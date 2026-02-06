import numpy as np
from scipy.signal import savgol_filter

def savitzky_golay(hsi_data, window_length=7, polyorder=2):
    """
    Savitzky-Golay 平滑
    :param hsi_data: 输入的高光谱数据，形状为 (samples, bands, rows, cols)
    :param window_length: 滑窗长度（必须为奇数）
    :param polyorder: 多项式阶数
    :return: 平滑后的高光谱数据，形状与输入相同
    """
    # 获取数据维度
    samples, bands, rows, cols = hsi_data.shape
    print(hsi_data.shape)
    # 初始化平滑后的数据
    smoothed_data = np.zeros_like(hsi_data)
    # 遍历每个样本
    for sample in range(samples):
        for i in range(rows):
            for j in range(cols):
                # 获取像素点的光谱数据 (bands,)
                spectrum = hsi_data[sample, :, i, j]
                # 应用 Savitzky-Golay 平滑
                smoothed_spectrum = savgol_filter(spectrum, window_length, polyorder)
                # 保存平滑后的光谱数据
                smoothed_data[sample, :, i, j] = smoothed_spectrum
    return smoothed_data


# 加载 data.npy 文件
data = np.load("E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/HuaFen2/NumPy/val_data.npy")

# 对数据进行SG处理
sg_data = savitzky_golay(data)

# 保存SG后的数据
np.save("E:/LiuWendan/dataset/202505/data/ShuJuZengQiang/HuaFen2/NumPy/val_data_sg.npy", sg_data)