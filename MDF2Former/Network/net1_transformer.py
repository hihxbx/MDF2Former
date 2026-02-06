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
from collections import OrderedDict
from einops.layers.torch import Rearrange
from torchsummary import summary
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import clever_format
from thop import profile


class channel_att(nn.Module):
    def __init__(self, in_dim):
        super(channel_att, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        # print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x  # B*C*H*W
        return out


class space_att(nn.Module):
    def __init__(self, in_dim):
        super(space_att, self).__init__()
        self.chanel_in = in_dim

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

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

#FTM
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
        
        # self.att = nn.Sequential(space_att(), channel_att(dim))
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
        x = x.reshape(B, C, N).permute(0,2,1)
        # x = x.reshape(B, N, C)
        return x

#FourierMixer
class FourierAttention(nn.Module):
    def __init__(self, dim, num_heads, h, w, flag,
                    qkv_bias = False,
                    attn_drop = 0.,
                    proj_drop = 0.,
                    method = 'dw_bn',  # 指定用于构建Q,K,V投影层的方法
                    kernel_size = 3,
                    stride_kv = 1,  # 对K、V进行投影时的卷积步长
                    stride_q = 1,  # 对Q进行投影时的卷积步长
                    padding_kv = 1,  # 对K、V进行投影时的卷积填充
                    padding_q = 1
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim=dim
        self.num_heads = num_heads
        self.scale = dim**-0.5

        self.fft = F3Filter(dim, h=h, w=w, flag=flag)
        self.norm = nn.LayerNorm(dim)
        # 使用_build_projection方法构建Q、K、V的投影层
        self.conv_proj_q = self._build_projection(
            dim, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim, kernel_size, padding_kv,
            stride_kv, method
        )

        # 使用nn.Linear构建Q、K、V的线性投影层
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
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

    def _build_projection(self,
                          dim,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn': # 深度可分离卷积+批归一化+重排
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim
                )),
                ('bn', nn.BatchNorm2d(dim)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj


    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
        # print('B, N, C:',x.shape)

        x_fft = self.fft(x) #FTM
        x=x_fft.reshape(B,H,W,C).permute(0,3,1,2)
        q = self.conv_proj_q(x)
        k = self.conv_proj_k(x)
        v = self.conv_proj_v(x)

        # (batch_size, num_heads, num_patches, embed_dim_per_head)，t=N，h*d=C
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        # 使用torch.einsum计算查询（q）和键（k）的点积注意力分数
        # 'bhlk,bhtk->bhlt'表示对batch中的每个头（h）和每个查询/键对（l,k和t,k）进行点积
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        # 使用torch.einsum计算加权和，将注意力权重（attn）应用于值（v）
        # 'bhlt,bhtv->bhlv'表示对每个头（h）的每个查询（l）和所有值（t,v）进行加权求和
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = x_fft + self.proj_drop(x)

        return x


# Fourierformer encoder block
# input(1,81,16)-->output(1,81,16)
class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_heads=2, h=14, w=8, init_values=1e-5, flag='Learn'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FourierAttention(dim, num_heads, h=h, w=w, flag='flag')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.attn(self.norm1(x)))))

        return x


class transformer(nn.Module):
    '''
    :param
        img_size: 图像尺寸
        in_chans: 输入的通道数
        num_classes: 分类类别数
        embed_dim: 每个patch的编码长度
        depth: token to token处理的深度
        drop_path_rate: 防过拟合的比例
        mlp_ratio: 多层感知器中隐藏层结点数目与输入结点数目的比例
        drop_rate:
'''
    def __init__(self, 
        img_size=13, 
        in_chans=30, 
        num_classes=16, 
        embed_dim=[32, 32, 32],
        depth=[2,2,2],
        mlp_ratio=[1,1,1], 
        num_heads=[2,4,8],
        uniform_drop=False,
        drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=None, 
        num_stages=3,
        dropcls=0,
        flag='GAP'
    ):
        super().__init__()
        self.name = 'MyNetwork'
        self.num_classes = num_classes
        self.num_stages = num_stages
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        sizes = [] #[13,11,9]
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
        patch_embed = PatchEmbed(img_size=img_size, patch_size=1, in_chans=128, embed_dim=embed_dim[0])
        self.patch_embed.append(patch_embed)

        for i in range(num_stages-1):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i+1])
            self.patch_embed.append(patch_embed)

        # 创建drop_path_rate:dpr[]
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)

        #循环构建Block
        cur = 0
        
        for i in range(num_stages):
            h = sizes[i]
            w = sizes[i]
            
            blocks = nn.ModuleList([BlockLayerScale(
                dim=embed_dim[i], 
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer, 
                num_heads=num_heads[i],
                h=h, 
                w=w,
                flag=flag)
                for j in range(depth[i])
            ])
        
            norm = norm_layer(embed_dim[i])
            cur += depth[i]

            # 根据循环变量名来创建和设置属性
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity() #占位符


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
        x = self.conv3d_features(x)
        x = rearrange(x, 'b n c h w -> b (n c) h w')
        x = self.conv2d_features(x)
        x = self.att(x)



        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x = self.patch_embed[i](x)
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
    net = transformer(img_size=9,
                        in_chans=15,
                        num_classes=15, 
                        embed_dim=[16,32,64],
                        depth=[2,1,1],
                        mlp_ratio=[1,1,1], 
                        num_heads=[2,4,8],
                        uniform_drop=False,
                        drop_rate=0., 
                        drop_path_rate=0., 
                        norm_layer=None, 
                        num_stages=3,
                        dropcls=0.3,
                        flag='GAP')


    a = torch.randn(1, 15, 9, 9)
    flops, params = profile(net, inputs=(a,))
    flops, params = clever_format([flops, params], '%.3f')
    print('Params: ',params)
    print('Flops: ',flops)
    y=net(a)
    print(y.shape)


