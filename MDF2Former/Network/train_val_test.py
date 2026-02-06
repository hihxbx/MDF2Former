import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from osgeo import gdal
from torch.utils.data import DataLoader, TensorDataset, random_split
# from net1 import fourierformer_p_v7
from net1_tztq_attention_transformer import transformer
from net1_tzrh_transformer import MyNetwork
# from net1_transformer import transformer


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')

# 加载数据
data = np.load('D:/创面细菌感染成像项目/代码/lwd/dataset/data_sg_pca.npy')
labels = np.load('D:/创面细菌感染成像项目/代码/lwd/dataset/labels.npy')
# 转换为PyTorch张量
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
# data_tensor = data_tensor.permute(0, 2, 3, 1)
# 数据集和数据加载器
dataset = TensorDataset(data_tensor, labels_tensor)

# 70%的数据用于训练，10%用于验证，20%用于测试
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, temp_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 定义模型
num_classes = len(np.unique(labels))  # 获取唯一标签的数量作为类别数
model = transformer(img_size=10,
                   in_chans=100,
                   num_classes=num_classes,
                   embed_dim=[16, 32, 64],
                   depth=[2, 1, 1],
                   mlp_ratio=[1, 1, 1],
                   num_heads=[2, 4, 8],
                   uniform_drop=False,
                   drop_rate=0.,
                   drop_path_rate=0.,
                   norm_layer=None,
                   num_stages=3,
                   dropcls=0.3,
                   flag='GAP').to(device)
# data = data_tensor.to(device)
# labels = labels_tensor.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
best_model_path = 'pth/best_model.pth'
best_f1_score = 0.0
print('----------开始训练----------')
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 将输入和标签移动到GPU（如果可用）
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}')


     # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_true_labels = []
    all_predicted_labels = []
    with torch.no_grad():
         for inputs, labels in val_loader:
             val_inputs, val_labels = inputs.to(device), labels.to(device)
             outputs = model(val_inputs)
             loss = criterion(outputs, val_labels)
             val_loss += loss.item()
             _, predicted = torch.max(outputs.data, 1)
             total += val_labels.size(0)
             correct += (predicted == val_labels).sum().item()
             # 收集真实标签和预测标签用于计算F1分数
             all_true_labels.extend(val_labels.cpu().numpy())
             all_predicted_labels.extend(predicted.cpu().numpy())

         val_loss /= len(val_loader)
         print(f'Validation Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')

         # 计算F1分数并更新最佳模型
         current_f1_score = f1_score(all_true_labels, all_predicted_labels, average='weighted')
         if current_f1_score > best_f1_score:
             best_f1_score = current_f1_score
             torch.save(model.state_dict(), best_model_path)  # 保存最佳模型状态
             print(f'Best model updated with F1 Score: {best_f1_score}')

print(f'Best model with F1 Score: {best_f1_score}')

# 加载模型
model.load_state_dict(torch.load('pth/best_model.pth',map_location=device,weights_only=True))  # 加载模型参数
model.to(device)
# 测试模型
model.eval()
correct = 0
total = 0
all_true_labels = []
all_predicted_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        test_inputs, test_labels = inputs.to(device), labels.to(device)
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
        # 收集真实标签和预测标签
        all_true_labels.extend(test_labels.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())

# 计算评估指标
accuracy = correct / total
precision = precision_score(all_true_labels, all_predicted_labels, average='weighted',zero_division=0)
recall = recall_score(all_true_labels, all_predicted_labels, average='weighted')
f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')


print('----------------测试集的模型评估指标结果-----------------')
print("Accuracy on test set: {:.4f}".format(accuracy))
print("Precision on test set: {:.4f}".format(precision))
print("Recall on test set: {:.4f}".format(recall))
print("F1 Score on test set: {:.4f}".format(f1))

# 输出分类报告
print(classification_report(all_true_labels, all_predicted_labels,digits=4))
    # cm = confusion_matrix(true_labels, predicted_labels)
    # normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
    # # 输出混淆矩阵
    # cm_matrix = pd.DataFrame(data=normalized_cm, index=range(24), columns=range(24))
    # plt.figure(figsize=(10, 10),dpi=100)
    # sns.heatmap(cm_matrix, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5,annot_kws={"fontsize":6},cbar_kws={'format':'%.0f%%'})
    # plt.xlabel('Predicted Label',family='Times New Roman')
    # plt.ylabel('True Label',family='Times New Roman')
    # plt.title('Confusion Matrix',family='Times New Roman')
    # plt.show()