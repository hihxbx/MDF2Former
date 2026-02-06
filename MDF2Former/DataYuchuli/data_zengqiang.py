import os
import numpy as np
import rasterio
from collections import defaultdict
from tqdm import tqdm

def parse_filename(file, parent_dir):
    """解析文件名，提取编号、浓度（格式：{parent_dir}{num}-{conc}_{count}.tif）"""
    prefix = parent_dir
    if not file.startswith(prefix):
        raise ValueError(f"文件 {file} 不符合 {parent_dir} 前缀命名规则！")
    file_base = file[len(prefix):]  # 去除目录名前缀
    num_part, rest = file_base.split('-')
    conc_part, _ = rest.split('_')   # 数量不影响分组，暂不解析
    return int(num_part), int(conc_part)

# ---------------------- 手动实现空间变换函数 ----------------------
def horizontal_flip(image_hwc):
    """水平翻转：(H, W, C) → 左右翻转"""
    return image_hwc[:, ::-1, :]


def vertical_flip(image_hwc):
    """垂直翻转：(H, W, C) → 上下翻转"""
    return image_hwc[::-1, :, :]


def rotate_90(image_hwc, k=1):
    """旋转90°倍数（k=1→90°, k=2→180°, k=3→270°）"""
    return np.rot90(image_hwc, k=k)


def augment_hyperspectral_tifs(
        original_dir,
        save_dir,
        target_num=5000
):
    """
    按(编号, 浓度)分组增强，确保覆盖所有编号(1-6)和浓度(4/5/7/9)
    :param original_dir: 原始.tif文件目录
    :param save_dir: 增强后文件保存目录
    :param target_num: 目标总样本数（原始+增强）
    """
    # 初始化保存目录
    os.makedirs(save_dir, exist_ok=True)
    parent_dir = os.path.basename(original_dir)  # 当前文件夹名称（用于解析文件名）
    original_files = [f for f in os.listdir(original_dir) if f.endswith('.tif')]
    num_original = len(original_files)

    # 原始样本数已满足目标，直接返回
    if num_original >= target_num:
        print("原始样本数已满足目标，无需增强！")
        return

     # 按 (编号, 浓度) 分组，确保每个类别组合都有增强
    groups = defaultdict(list)
    for f in original_files:
        num, conc = parse_filename(f, parent_dir)
        groups[(num, conc)].append(f)
    num_groups = len(groups)
    total_aug_needed = target_num - num_original  # 需要增强的总样本数

    # 分配每组增强数量（平均分配+处理余数，确保所有组都有增强）
    avg_aug_per_group = total_aug_needed // num_groups
    remaining = total_aug_needed % num_groups
    group_aug_counts = {
        group: avg_aug_per_group + (1 if i < remaining else 0)
        for i, group in enumerate(groups.keys())
    }

    # 定义空间变换列表（名称+函数），模拟临床空间采集偏差
    transform_options = [
        ("horizontal_flip", horizontal_flip),
        ("vertical_flip", vertical_flip),
        ("rotate_90", lambda x: rotate_90(x, k=1)),
        ("rotate_180", lambda x: rotate_90(x, k=2)),
        ("rotate_270", lambda x: rotate_90(x, k=3)),
    ]
    num_transforms = len(transform_options)

    aug_count = 0  # 已生产的增强样本数
    # 1. 先保存所有原始样本（避免增强时遗漏）
    for f in tqdm(original_files, desc="Saving original files"):
        original_save_path = os.path.join(save_dir, f"original_{f}")
        if not os.path.exists(original_save_path):
            file_path = os.path.join(original_dir, f)
            with rasterio.open(file_path) as src:
                img = src.read()
                profile = src.profile
                with rasterio.open(original_save_path, 'w', **profile) as dst:
                    dst.write(img)

    # 2. 按组生成增强样本
    for (num, conc), group_files in tqdm(groups.items(), desc="Processing groups"):
        needed_aug = group_aug_counts[(num, conc)]
        if needed_aug <= 0:
            continue  # 该组无需增强

        generated = 0  # 该组已生成的增强样本数
        while generated < needed_aug and aug_count < total_aug_needed:
            # 随机选组内一个文件进行增强
            f = np.random.choice(group_files)
            file_path = os.path.join(original_dir, f)
            with rasterio.open(file_path) as src:
                img_chw = src.read()  # 原始格式：(C, H, W)
                profile = src.profile  # 元数据（投影、分辨率等）
                C, H, W = img_chw.shape

                # 步骤1：(C, H, W) → (H, W, C)（适配空间变换）
                img_hwc = img_chw.transpose(1, 2, 0)

                # 步骤2：随机选择空间变换
                transform_name, transform_func = transform_options[np.random.randint(num_transforms)]
                img_hwc_aug = transform_func(img_hwc)

                # 步骤3：(H, W, C) → (C, H_new, W_new)（还原高光谱格式）
                img_chw_aug = img_hwc_aug.transpose(2, 0, 1)

                # 检查增强是否有效（避免生成与原图一致的样本）
                if np.array_equal(img_chw_aug, img_chw):
                    continue

                # 动态更新元数据（高度、宽度适配增强后图像）
                new_H, new_W = img_hwc_aug.shape[:2]
                new_profile = profile.copy()
                new_profile['height'] = new_H
                new_profile['width'] = new_W
                new_profile['count'] = C  # 空间变换不改变波段数

                # 保存增强样本（命名格式：aug_编号-浓度_序号_原文件名.tif）
                aug_filename = f"aug_{generated + 1}_{f}"
                aug_save_path = os.path.join(save_dir, aug_filename)
                with rasterio.open(aug_save_path, 'w', **new_profile) as dst:
                    dst.write(img_chw_aug)

                generated += 1
                aug_count += 1

    total_final = num_original + aug_count
    print(f"增强完成！总样本数：{total_final}（原始{num_original} + 增强{aug_count}）")


if __name__ == "__main__":
    # 配置路径
    ORIGINAL_TIF_DIR = "E:/LiuWendan/dataset/202505/10_×_10/FY"  # 原始.tif文件目录
    AUGMENTED_TIF_DIR = "E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/YS2/FY"  # 增强后保存目录
    # ORIGINAL_TIF_DIR = "E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/test/FY"  # 原始.tif文件目录
    # AUGMENTED_TIF_DIR = "E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/test/2"  # 增强后保存目录

    # 执行增强
    augment_hyperspectral_tifs(
        original_dir=ORIGINAL_TIF_DIR,
        save_dir=AUGMENTED_TIF_DIR,
        target_num=5886
    )