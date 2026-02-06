import numpy as np
import os
import rasterio
from osgeo import gdal
from osgeo import gdal_array


white = gdal.Open('E:/LiuWendan/dataset/JiaoZhun/white_10×10_1.tif')
black= gdal.Open('E:/LiuWendan/dataset/JiaoZhun/white_10×10_1.tif')

width = white.RasterXSize
height = white.RasterYSize
bands = white.RasterCount

white_Array = np.array(white.ReadAsArray(0, 0, width, height)).astype("float64")

Bacterium = ['BX', 'DC', 'FS', 'TL','BM', 'BP', 'FY', 'JH']

for bacterium in Bacterium:
            # 输入文件夹路径
            input_folder = f"E:/LiuWendan/dataset/202505/10_×_10/未校准/{bacterium}"
            # 输出文件夹路径
            output_folder = f"E:/LiuWendan/dataset/202505/10_×_10/校准/{bacterium}"

            # 如果输入文件夹不存在，则跳过
            if not os.path.exists(input_folder):
                continue
            # 如果输出文件夹不存在，则创建
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 获取文件夹下所有TIFF文件的路径
            tif_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.tif')]

            # 逐个处理每个TIFF文件
            for tif_file in tif_files:
                with rasterio.open(tif_file) as src:
                    img_Array = np.array(src.read()).astype("float64")
                    img_bands, img_height, img_width = img_Array.shape

                    # 调整校准图像尺寸以匹配当前图像
                    if (img_height, img_width) != (height, width):
                        print(
                            f"警告: {tif_file} 尺寸({img_width},{img_height})与校准图像({white_width},{white_height})不一致，调整校准图像")
                        # 调整white_Array尺寸（简单方法：裁剪或填充）
                        white_resized = white_Array[:, :img_height, :img_width]
                    else:
                        white_resized = white_Array
                img_new = img_Array[:, :height, :] / (white_Array + 1e-8)

                gdal_array.SaveArray(img_new, output_folder + f"/{os.path.splitext(os.path.basename(tif_file))[0]}.tif", format="GTiff")

print(f"{bacterium} 处理完成，共处理 {len(tif_files)} 个文件\n")