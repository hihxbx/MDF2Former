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

        # 按「编号-浓度」类别收集文件（适配original_/aug_前缀）
        category_files = defaultdict(list)
        for file in os.listdir(bacteria_dir):
            if file.endswith(".tif"):
                # 步骤1：移除.tif后缀，处理前缀
                file_base = file[:-4]  # 如 "original_YS1-4_1.tif" → 移除.tif后为 "original_YS1-4_1"

                # 步骤2：剥离original_前缀
                if file_base.startswith("original_"):
                    stripped = file_base[len("original_"):]  # 如 "YS1-4_1"
                # 步骤3：剥离aug_前缀（格式：aug_增强量_细菌名编号-浓度_数量）
                elif file_base.startswith("aug_"):
                    parts = file_base.split("_", 2)  # 分割为 ["aug", "增强量", "细菌名编号-浓度_数量"]
                    stripped = parts[2] if len(parts) >= 3 else ""  # 取第三部分（如 "YS2-5_2"）
                else:
                    stripped = file_base  # 其他前缀（理论无，仅容错）

                # 步骤4：提取「细菌名编号-浓度」（移除数量后缀）
                category_part = stripped.rsplit("_", 1)[0]  # 如 "YS1-4" 或 "YS2-5"

                # 步骤5：提取「编号-浓度」（移除细菌名前缀，如YS）
                bacteria_prefix = bacteria
                if category_part.startswith(bacteria_prefix):
                    num_conc_part = category_part[len(bacteria_prefix):]  # 如 "1-4" 或 "2-5"
                    category = num_conc_part
                else:
                    category = "unknown"  # 命名不规范时的容错标记
                    print(f"警告：{bacteria} 文件夹下 {file} 命名不规范，类别暂标记为unknown")

                category_files[category].append(file)

    # 划分每个类别的文件
    for category, files in category_files.items():
        if category == "unknown":
            print(f"跳过不规范类别 {category} 的文件划分")
            continue
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