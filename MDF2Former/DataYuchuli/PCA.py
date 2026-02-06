import numpy as np
from sklearn.decomposition import PCA


def applyPCA(input_file, components, output_file):
        # 加载.npy格式的数据
        data = np.load(input_file) #(16192,300,10,10)
        data = np.transpose(data,(0,2,3,1)) #(16192,10,10,300)

        # 验证数据形状
        n_samples, height, width, channels = data.shape
        assert data.shape[1] == 10 and data.shape[2] == 10 and data.shape[3] == 300, \
            "Input data must have shape (n_samples, 10, 10, 300)"

        # 将数据重塑为二维数组 (n_samples * 10 * 10, 300)
        reshaped_data = data.reshape(-1, channels)

        # 应用PCA
        pca = PCA(n_components=components, whiten=True)
        transformed_data = pca.fit_transform(reshaped_data)

        # 将数据重塑回原始的三维形状，但最后一个维度变为components
        transformed_data = transformed_data.reshape(n_samples, height, width, components)
        transformed_data = np.transpose(transformed_data,(0,3,1,2)) #(16192,100,10,10)

        # 保存变换后的数据到.npy文件
        np.save(output_file, transformed_data)

        return transformed_data


# 使用示例
if __name__ == "__main__":

    input_data_file = 'E:/LiuWendan/dataset/202505/data/ChuShiHuaFen/NumPy/val_data_sg.npy'  # 输入文件路径
    output_data_file = 'E:/LiuWendan/dataset/202505/data/ChuShiHuaFen/NumPy/val_data_sg_pca100.npy'  # 输出文件路径
    num_components = 100  # 主成分数量

    transformed_data = applyPCA(input_data_file, num_components, output_data_file)
