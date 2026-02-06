import os
import random
import shutil
from collections import defaultdict

# 设置随机种子以确保结果可复现
random.seed(42)

# 源数据目录路径
data_dir = "E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/YS2"

# 目标目录路径
output_dir = "E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/HuaFen2"

# 创建train、test、val目录
for split in ["train", "test", "val"]:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# 获取所有细菌文件夹名
bacteria_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# 处理每个细菌文件夹
for bacteria in bacteria_folders:
    bacteria_dir = os.path.join(data_dir, bacteria)
    print(f"处理细菌文件夹: {bacteria}")

    # 为train、test、val创建对应的细菌文件夹
    for split in ["train", "test", "val"]:
        split_bacteria_dir = os.path.join(output_dir, split, bacteria)
        os.makedirs(split_bacteria_dir, exist_ok=True)

    # 按类别收集文件
    category_files = defaultdict(list)
    for file in os.listdir(bacteria_dir):
        if file.endswith(".tif"):
            # 提取类别信息（如"1-4"）
            parts = file.split("_")[0].split("-")
            if len(parts) >= 2:
                category = f"{parts[0]}-{parts[1]}"
                category_files[category].append(file)

    # 划分每个类别的文件
    for category, files in category_files.items():
        # 随机打乱文件列表
        random.shuffle(files)
        total = len(files)

        # 计算划分数量
        train_size = int(total * 0.7)
        test_size = int(total * 0.2)
        val_size = total - train_size - test_size

        # 划分文件
        train_files = files[:train_size]
        test_files = files[train_size:train_size + test_size]
        val_files = files[train_size + test_size:]

        # 复制文件到对应的目录
        for split, split_files in zip(["train", "test", "val"], [train_files, test_files, val_files]):
            for file in split_files:
                src_path = os.path.join(bacteria_dir, file)
                dst_path = os.path.join(output_dir, split, bacteria, file)
                shutil.copy2(src_path, dst_path)  # 保留文件元数据

        print(
            f"类别 {category} 已划分为: 训练集({len(train_files)}), 测试集({len(test_files)}), 验证集({len(val_files)})")

print("数据划分完成！")