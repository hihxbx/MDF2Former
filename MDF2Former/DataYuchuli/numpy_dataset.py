import os
import numpy as np
from osgeo import gdal

path = 'E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/HuaFen2/test'
data = []
labels = []
label_map = {}  # 二级目录名 → 标签ID的映射
current_label = 0

# 遍历细菌文件夹并分配标签
for second_dir in os.listdir(path):
    second_dir_path = os.path.join(path, second_dir)

    # 为每个二级目录分配唯一标签
    if second_dir not in label_map:
        label_map[second_dir] = current_label
        current_label += 1

    print(f"处理二级目录: {second_dir_path}, 标签: {label_map[second_dir]}")

    # 直接遍历二级目录下的所有文件
    for filename in os.listdir(second_dir_path):
        if not filename.endswith(('.tif', '.tiff')):
            continue

        file = os.path.join(second_dir_path, filename)
        try:
            # 打开图像
            img = gdal.Open(file)
            if img is None:
                raise Exception(f"无法打开文件: {file}")

            # 读取图像数据：从图像原点开始读取一个10×10大小的区域并转换为NumPy数组
            img_data = img.ReadAsArray(0, 0, 10, 10)

            # 添加到数据集，使用二级目录的标签
            data.append(img_data)
            labels.append(label_map[second_dir])

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

# 转换为NumPy数组
data = np.array(data)
labels = np.array(labels)

# 保存数据
output_dir = 'E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/HuaFen2/NumPy'
np.save(os.path.join(output_dir, 'test_data.npy'), data)
np.save(os.path.join(output_dir, 'test_labels.npy'), labels)

# 保存标签映射
with open(os.path.join(output_dir, 'test_label_map.txt'), 'w') as f:
    for key, value in label_map.items():
        f.write(f"{key}: {value}\n")

print(f"数据保存完成: {data.shape}, 标签: {np.unique(labels)}")
print(f"标签映射: {label_map}")