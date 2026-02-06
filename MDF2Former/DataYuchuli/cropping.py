from osgeo import gdal
from osgeo import gdal_array
import os

##############################################################################################
"""每次裁剪前，需要修改的一些参数"""
light_source = "近红外"        #光源(需要改这个):"可见" "近红外" "紫光"
# concentration = 10           #浓度(需要改这个)
folder = 'TL'               #细菌名(需要改这个):'BP' 'BM' 'DC' 'FC' 'FS' 'FY' 'JH' 'LN' 'PT' 'YX'
kernel_size = 10            #裁剪大小
stride = 10                 #步长
time=0
##############################################################################################

input_folder = f"E:/LiuWendan/dataset/202505/any_×_any/{folder}"
output_base = f"E:/LiuWendan/dataset/202505/{kernel_size}_×_{kernel_size}/{folder}"

tif_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.tif')]
for tif_file in tif_files:
    img = gdal.Open(tif_file)
    width = img.RasterXSize
    height = img.RasterYSize #获取图像的高
    bands = img.RasterCount #获取波段数


    # if os.path.exists(f"E:/LiuWendan/dataset/202505/{kernel_size}_×_{kernel_size}/{folder}/{os.path.splitext(os.path.basename(tif_file))[0]}"):  # 检查是否已经创建用于保存10×10图像的目录
    #     print("已有相关文件目录")
    # else:
    #     os.makedirs(f"D:/创面细菌感染成像项目/大鼠实验/20240811/{kernel_size}_×_{kernel_size}/{time}/{os.path.splitext(os.path.basename(tif_file))[0]}")  # 创建目录保存裁剪的10×10图像
    if os.path.exists(
            f"E:/LiuWendan/dataset/202505/{kernel_size}_×_{kernel_size}/{folder}"):  # 检查是否已经创建用于保存10×10图像的目录
            print("已有相关文件目录")
    else:
        os.makedirs(f"E:/LiuWendan/dataset/202505/{kernel_size}_×_{kernel_size}/{folder}")  # 创建目录保存裁剪的10×10图像

    base_name = os.path.splitext(os.path.basename(tif_file))[0]
    or_num = 0

    for x in range(0, width - kernel_size + 1, stride):
        for y in range(0, height - kernel_size + 1, stride):
    # 读取图像数据
            if (x > width - kernel_size) or (y > height - kernel_size): continue
            else:
                or_num = or_num + 1
                img_data = img.ReadAsArray(x, y, kernel_size, kernel_size)

                output_path=os.path.join(output_base, f"{base_name}_{or_num}.tif")
                gdal_array.SaveArray(img_data, output_path, format="GTiff")

    print(f"{base_name}裁剪{kernel_size}_×_{kernel_size}子图共计{or_num}张！\n",)
