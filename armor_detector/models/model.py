import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from PIL import Image

# --- 1. 超参数和配置 (保持不变) ---
DATASET_PATH = "/home/heish/Documents/Projects/XJTU-RMV-Task05/armor_detector/src/armor_detector/models/datasets"
MODEL_SAVE_PATH = "/home/heish/Documents/Projects/XJTU-RMV-Task05/armor_detector/src/armor_detector/models/best_digit_model.pth"
ONNX_EXPORT_PATH = "/home/heish/Documents/Projects/XJTU-RMV-Task05/armor_detector/src/armor_detector/models/model/armor_digit.onnx"

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
# INPUT_SIZE = 28 * 20 # CNN不再需要这个参数
NUM_CLASSES = 9
VALIDATION_SPLIT = 0.2

# --- 2. 数据预处理 (保持不变) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}

# --- 3. 加载并切分数据集 (保持不变) ---
print("正在从单个目录加载数据集...")
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=None)

num_data = len(full_dataset)
val_size = int(VALIDATION_SPLIT * num_data)
train_size = num_data - val_size

train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

train_subset.dataset.transform = data_transforms['train']
val_subset.dataset.transform = data_transforms['val']

dataloaders = {
    'train': DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}
class_names = full_dataset.classes

print(f"总样本数: {num_data}")
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"找到的类别 (按字母顺序): {class_names}")
print(f"类别到索引的映射: {full_dataset.class_to_idx}")


# ==================== 4. 定义CNN模型 (修改部分) ====================
# 用一个简单的卷积神经网络 (CNN) 替换掉原来的 MLP
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 输入形状: (N, 1, 28, 20)  N=批次大小, 1=灰度通道
        self.layer1 = nn.Sequential(
            # 卷积层1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # 经过卷积 -> (N, 16, 28, 20)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 池化层1
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 经过池化 -> (N, 16, 14, 10)
        )
        self.layer2 = nn.Sequential(
            # 卷积层2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 经过卷积 -> (N, 32, 14, 10)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 池化层2
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 经过池化 -> (N, 32, 7, 5)
        )
        # 全连接层
        # 将卷积层的输出拉平成一维向量，然后送入全连接层
        # 拉平后的维度 = 32 * 7 * 5 = 1120
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x 的原始形状是 (N, 1, 28, 20)
        out = self.layer1(x)
        out = self.layer2(out)
        # 在送入全连接层前，将特征图拉平
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ====================================================================


# ==================== 5. 训练过程 (修改部分) ====================
def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 修改: 实例化新的CNN模型 ---
    model = CNN(num_classes=NUM_CLASSES).to(device)
    # --------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # CNN的训练循环与MLP完全一样，无需修改
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"找到更优模型，已保存到 {MODEL_SAVE_PATH}")

    print(f'\n训练完成! 最佳验证集准确率: {best_val_acc:.4f}')

# ====================================================================


# ==================== 6. 导出到 ONNX (修改部分) ====================
def export_to_onnx():
    print(f"\n正在将最佳模型导出到 {ONNX_EXPORT_PATH}...")
    
    # --- 修改: 实例化新的CNN模型 ---
    model = CNN(num_classes=NUM_CLASSES)
    # --------------------------------

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # dummy_input 的形状 (1, 1, 28, 20) 对于CNN是完美的，无需修改
    dummy_input = torch.randn(1, 1, 28, 20) 
    
    # 导出函数的其他参数也无需修改
    torch.onnx.export(model, dummy_input, ONNX_EXPORT_PATH,
                      export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("ONNX 模型导出成功!")

# ====================================================================

if __name__ == '__main__':
    train_model()
    export_to_onnx()