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
    conc_part, _ = rest.split('_')  # 数量字段不影响分组，暂不解析
    return int(num_part), int(conc_part)


def delete_and_save(original_dir, save_dir, target_remain):
    """
    按(编号, 浓度)分组均衡删减高光谱.tif，剩余样本保存到新目录
    :param original_dir: 原始高光谱数据目录
    :param save_dir: 删减后数据保存目录
    :param target_remain: 目标剩余总样本数（需 ≤ 原始样本数）
    """
    # 初始化保存目录
    os.makedirs(save_dir, exist_ok=True)
    parent_dir = os.path.basename(original_dir)  # 当前文件夹名称（用于解析文件名）
    original_files = [f for f in os.listdir(original_dir) if f.endswith('.tif')]
    num_original = len(original_files)

    # 目标剩余数≥原始数时，直接复制所有文件
    if target_remain >= num_original:
        print(f"目标剩余数 {target_remain} ≥ 原始数 {num_original}，无需删减！")
        for f in tqdm(original_files, desc="Copying all files"):
            src_path = os.path.join(original_dir, f)
            dst_path = os.path.join(save_dir, f)
            with rasterio.open(src_path) as src:
                img = src.read()  # 读取高光谱图像（C, H, W）
                profile = src.profile  # 保留原始元数据（投影、分辨率等）
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    dst.write(img)
        return

    # 按 (编号, 浓度) 分组，统计每个组的文件列表
    groups = defaultdict(list)
    for f in original_files:
        num, conc = parse_filename(f, parent_dir)
        groups[(num, conc)].append(f)
    num_groups = len(groups)

    # 计算每个组需保留的样本数（按比例分配，保证总和=target_remain）
    total_original_per_group = {g: len(files) for g, files in groups.items()}
    total_original = sum(total_original_per_group.values())
    ratio = target_remain / total_original  # 全局保留比例

    # 先按比例分配基础保留数，再处理余数（确保总和精确=target_remain）
    base_keep = {g: int(len(files) * ratio) for g, files in groups.items()}
    remaining = target_remain - sum(base_keep.values())  # 分配后剩余的数量
    group_list = list(groups.keys())  # 按组顺序分配余数

    for i in range(remaining):
        group = group_list[i]
        base_keep[group] += 1  # 前`remaining`个组各多保留1个样本

    # 对每个组执行删减（随机保留指定数量的文件）
    for (num, conc), files in tqdm(groups.items(), desc="Processing groups"):
        keep_num = base_keep[(num, conc)]
        if keep_num <= 0:
            print(f"警告：组 ({num}, {conc}) 保留数量为0，可能导致该类别样本缺失！")
            continue

        # 随机选择`keep_num`个文件保留
        keep_files = np.random.choice(files, size=keep_num, replace=False)
        for f in keep_files:
            src_path = os.path.join(original_dir, f)
            dst_path = os.path.join(save_dir, f)
            with rasterio.open(src_path) as src:
                img = src.read()  # 读取高光谱图像（C, H, W）
                profile = src.profile  # 保留原始元数据
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    dst.write(img)

    print(f"删减完成！原始样本数 {num_original} → 剩余样本数 {target_remain}，保存至 {save_dir}")


if __name__ == "__main__":
    ORIGINAL_TIF_DIR = "E:/LiuWendan/dataset/202505/10_×_10/FS"  # 原始数据目录
    SAVE_DIR = "E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/FS"  # 删减后数据保存目录
    TARGET_REMAIN = 6801  # 目标剩余总样本数
    # ==========================================================

    delete_and_save(
        original_dir=ORIGINAL_TIF_DIR,
        save_dir=SAVE_DIR,
        target_remain=TARGET_REMAIN
    )