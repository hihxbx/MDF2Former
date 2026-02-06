import math
from functools import partial
from cv2 import sqrt
from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import PatchEmbed
import torch.fft
from torch.nn.modules.container import Sequential

from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import clever_format
from thop import profile
# from attontion import CAM_Module
# from .attontion import CAM_Module

class channel_att(nn.Module):
    def __init__(self, in_dim):
        super(channel_att, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) 
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x  #C*H*W
        return out
    
class space_att(nn.Module):
    def __init__(self, in_dim):
        super(space_att, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=13, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=5, stride=1, padding=1) 
        
        # self.proj = nn.Sequential(
        #     nn.Conv2d(dim_in, dim_out, kernel_size=5, stride=1, padding=1),
        #     nn.BatchNorm2d(dim_out),
        #     nn.GELU(),
        # )
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        # print('DownLayer x:',x.size())
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        # print('DownLayer x:',x.size())
        x = self.proj(x).permute(0, 2, 3, 1)
        # print('DownLayer x:',x.size())
        x = x.reshape(B, -1, self.dim_out)
        # print('DownLayer x:',x.size())
        return x

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.norm = LayerNorm(in_features, eps=1e-6, data_format="channels_first")
        # self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.pos = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        # self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        # self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        # B, N, C = x.shape
        # H = W = int(math.sqrt(N))
        # x = x.reshape(B,H,W,C).permute(0,3,1,2).contiguous() 
        # x = self.norm(x)
        # x = self.fc1(x)
        # x = self.act(x)
        # x = x + self.act(self.pos(x))
        # x = self.fc2(x)
        # x = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
#         self.pwc = eSEModule(in_channels,bias=True,stride = 1 )
    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

class F3Filter(nn.Module):
    def __init__(self, dim, h=14, w=8, flag='Learn'):
        super().__init__()
        self.name = 'F3Filter'
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w,  2, dtype=torch.float32) * 0.02)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
        )
        
        self.att = nn.Sequential(space_att(dim), channel_att(dim))
        self.pooling = nn.AdaptiveAvgPool2d((h,w))
        self.w = w
        self.h = h
        self.flag = flag
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C).permute(0,3,1,2)
        # print('x.view(B, a, b, C):',x.size())
        x = x.to(torch.float32)
        if self.flag == 'Learn':
            # print('flag is learn')
            x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
            weight = torch.view_as_complex(self.complex_weight)
            x = x * weight
            x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho')
        elif self.flag == 'GAP':
            # print('flag is Pooling')
            x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
            x = self.pooling(x.real) + self.pooling(x.imag) * 1j
            x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho')
       
        elif self.flag == 'ATT':
            # print('flag is SA')
            x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
            x = self.att(x.real) + self.att(x.imag) * 1j
            x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho') 
        else:
            # print('flag is conv')
            x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
            x = self.conv(x.real) + self.conv(x.imag) * 1j
            x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho')
        # print('x_fft:',x_fft.shape)
        # x = x.reshape(B, C, N).permute(0,2,1)
        x = x.reshape(B, N, C)
        return x

class FourierAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, h, w, flag):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.fft = F3Filter(dim, h=h, w=w, flag=flag)
        # self.fft = AFNO2D(hidden_size=dim, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)
        self.qkv = nn.Linear(dim, dim * 3)
        self.norm = nn.LayerNorm(dim)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv
        # self.gamma = nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True)
        # self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        # self.q = nn.Linear(dim, dim)
        # self.kv = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim * 2)
        # )
        # self.proj = nn.Linear(dim*3, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
        # print('B, N, C:',x.shape)
        # x_org = x
        # x_fft = x + self.proj_drop(self.fft(x))
        x_fft = self.fft(x)
        qkv = self.qkv(x_fft).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        
        q = qkv[0].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = qkv[1].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = qkv[2].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # print('q:',q.shape)
        # print('k:',k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                
        # v without partition
        v_lcm = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)
        lcm = self.get_v(v_lcm)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        
        # x = self.proj(torch.cat([x, x_org, lcm], dim=-1))
        x = x + lcm
        x = self.proj(x)
        x = x_fft + self.proj_drop(x)

        return x
    
class FourierBigConv(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, h, w, flag):
        super().__init__()
        
        self.fft = F3Filter(dim, h=h, w=w, flag=flag)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj_att = nn.Conv2d(dim, dim, 1)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
        x_fft = self.fft(x)
        x_fft = self.norm(x_fft)  
        x1 = x_fft.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
        a = self.a(x1)
        v = self.v(x1)
        att = a * v
        att = self.proj_att(att)
        att = att.permute(0, 2, 3, 1).contiguous().view(B, N, C)
                
        # v without partition
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        
        # x = self.proj(torch.cat([x, x_org, lcm], dim=-1))
        att = att + lcm
        att = self.proj(att)
        out = x_fft + self.proj_drop(att)

        return out


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_heads=2, sr_ratio=1, h=14, w=8, flag='GAP'):
        super().__init__()
        # print('dim:',dim)
        self.norm1 = norm_layer(dim)
        self.attn = FourierAttention(dim, num_heads, sr_ratio, h=h, w=w, flag=flag)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = PVT2FFN(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        # print('x:',x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(self.attn(self.norm1(x)))))
        return x

class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_heads=2, sr_ratio=1, h=14, w=8, init_values=1e-5, flag='Learn'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.filter = GlobalFilter(dim, h=h, w=w)
        self.attn = FourierAttention(dim, num_heads, sr_ratio, h=h, w=w, flag=flag)
        # self.attn = FourierBigConv(dim, num_heads, sr_ratio, h=h, w=w, flag=flag)
        # self.attn = FourierAttentionPerformer(dim, num_heads, sr_ratio, h=h, w=w, flag=flag)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = PVT2FFN(in_features=dim, hidden_features=mlp_hidden_dim)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.attn(self.norm1(x)))))
        # x = x + self.drop_path(self.mlp(self.norm2(self.attn(self.norm1(x)))))
        return x

#Pyramidal version
class fourierformer_p(nn.Module):
    def __init__(self, 
        img_size=13, 
        in_chans=30, 
        num_classes=16, 
        embed_dim=[32, 32, 32], 
        downsample = False,
        depth=[2,2,2],
        mlp_ratio=[1,1,1], 
        num_heads=[2,4,8], 
        sr_ratio=[4,2,1], 
        uniform_drop=False,
        drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=None, 
        num_stages=3,
        dropcls=0,
        flag='GAP'
    ):
        super().__init__()
        self.name = 'fourierformer_p'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.depths = depth
        # self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*in_chans, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # self.speatt = CAM_Module(in_chans)
        
        self.patch_embed = nn.ModuleList()
        patch_embed = PatchEmbed(
                img_size=img_size, patch_size=1, in_chans=64, embed_dim=embed_dim[0])
        num_patches = patch_embed.num_patches
        self.patch_embed.append(patch_embed)

        for i in range(num_stages-1):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i+1])
            self.patch_embed.append(patch_embed)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0
        
        for i in range(num_stages):
            h = sizes[i]
            w = h // 2 + 1
            
            blocks = nn.ModuleList([BlockLayerScale(
                dim=embed_dim[i], 
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer, 
                num_heads=num_heads[i], 
                sr_ratio=sr_ratio[i], 
                h=h, 
                w=w,
                flag=flag)
                for j in range(depth[i])])
        
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            # setattr(self,f"downsample{i + 1}", downsamplelayers)if i>0 else
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)


        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        # Classifier head
        # self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = x.unsqueeze(1)
        x = rearrange(x, 'b n h w c -> b n c h w')
        # x = self.speatt(x)
        # x = rearrange(x, 'b c h w n -> b n c h w')
        # print('x:',x.size())
        x = self.conv3d_features(x)
        # print('conv3d_features:',x.size())
        x = rearrange(x, 'b n c h w -> b (n c) h w')
        # print('rearrange:',x.size())
        x = self.conv2d_features(x)
        x = self.patch_embed[0](x)

        x0 = x
        block0 = getattr(self, f"block{1}")
        norm0 = getattr(self, f"norm{1}")
        for blk in block0:
            x0 = blk(x0)
            x0 = norm0(x0)
            
        x1 = x0
        x1 = self.patch_embed[1](x1)
        block1 = getattr(self, f"block{2}")
        norm1 = getattr(self, f"norm{2}")
        for blk in block1:
            x1 = blk(x1)
            x1 = norm1(x1)
            
        x2 = x1
        x2 = self.patch_embed[2](x2)
        block2 = getattr(self, f"block{3}")
        norm2 = getattr(self, f"norm{3}")
        for blk in block2:
            x2 = blk(x2)
            x2 = norm2(x2)

        return x2.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x

class fourierformer_p_v7(nn.Module):
    def __init__(self, 
        img_size=13, 
        in_chans=30, 
        num_classes=16, 
        embed_dim=[32, 32, 32], 
        downsample = False,
        depth=[2,2,2],
        mlp_ratio=[1,1,1], 
        num_heads=[2,4,8], 
        sr_ratio=[4,2,1], 
        uniform_drop=False,
        drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=None, 
        num_stages=3,
        dropcls=0,
        flag='GAP'
    ):
        super().__init__()
        self.name = 'fourierformer_p_v7'
        self.num_classes = num_classes
        self.num_stages = num_stages
        # self.depths = depth
        # self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=16*in_chans, out_channels=128, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        self.att = nn.Sequential(space_att(128),
                                 channel_att(128))
        self.patch_embed = nn.ModuleList()
        patch_embed = PatchEmbed(
                img_size=img_size, patch_size=1, in_chans=128, embed_dim=embed_dim[0])
        # num_patches = patch_embed.num_patches
        self.patch_embed.append(patch_embed)

        for i in range(num_stages-1):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i+1])
            self.patch_embed.append(patch_embed)

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0
        
        for i in range(num_stages):
            h = sizes[i]
            w = h // 2 + 1
            
            blocks = nn.ModuleList([BlockLayerScale(
                dim=embed_dim[i], 
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer, 
                num_heads=num_heads[i], 
                sr_ratio=sr_ratio[i], 
                h=h, 
                w=w,
                flag=flag)
                for j in range(depth[i])])
        
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            # setattr(self,f"downsample{i + 1}", downsamplelayers)if i>0 else
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = x.unsqueeze(1)
        # x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b n c h w -> b (n c) h w')
        x = self.conv2d_features(x)
        x = self.att(x)
        
        for i in range(3):
            # patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x = self.patch_embed[i](x)  # s = feature map size after patch embedding
            for blk in block:
                x = blk(x)
            x = norm(x)

        return x.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x



if __name__ == '__main__':
    net = fourierformer_p_v7(img_size=9,
                        in_chans=15,
                        num_classes=15, 
                        embed_dim=[16,32,64], 
                        downsample = False,
                        depth=[2,1,1],
                        mlp_ratio=[1,1,1], 
                        num_heads=[2,4,8], 
                        sr_ratio=[1,1,1], 
                        # representation_size=None, 
                        uniform_drop=False,
                        drop_rate=0., 
                        drop_path_rate=0., 
                        norm_layer=None, 
                        num_stages=3,
                        dropcls=0.3,
                        flag='GAP').cuda()

    # print(net)
    a = torch.randn(1, 15, 9, 9).cuda()
    flops, params = profile(net, inputs=(a,))
    flops, params = clever_format([flops, params], '%.3f')

    print('Params: ',params)
    print('Flops: ',flops)
