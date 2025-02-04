import torch
from torchvision.models import resnet50

# 加载预训练的 ResNet50 模型
model = resnet50(pretrained=True)

# 如果不需要预训练权重，可以设置 pretrained=False
# model = resnet50(pretrained=False)

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 假设 data_loader 和 valid_data_loader 是自定义的函数，需要根据实际情况实现
# 这里仅提供一个示例框架
def data_loader(datadir, batch_size, mode='train'):
    # 实现数据加载逻辑，返回一个 DataLoader
    # 示例：使用自定义的 Dataset 类
    dataset = CustomDataset(datadir, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def valid_data_loader(datadir, csv_file):
    # 实现验证数据加载逻辑，返回一个 DataLoader
    dataset = CustomValidDataset(datadir, csv_file)
    return DataLoader(dataset, batch_size=10, shuffle=False)

class Runner(object):
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # 记录全局最优指标
        self.best_acc = 0

    # 定义训练过程
    def train_pm(self, train_datadir, val_datadir, **kwargs):
        print('start training ... ')
        self.model.train()

        num_epochs = kwargs.get('num_epochs', 0)
        csv_file = kwargs.get('csv_file', None)
        save_path = kwargs.get("save_path", "/home/aistudio/output/")

        # 定义数据读取器，训练数据读取器
        train_loader = data_loader(train_datadir, batch_size=10, mode='train')

        for epoch in range(num_epochs):
            for batch_id, (img, label) in enumerate(train_loader):
                # 将数据移动到设备上（CPU 或 GPU）
                img, label = img.to(self.model.device), label.to(self.model.device)
                # 运行模型前向计算，得到预测值
                logits = self.model(img)
                avg_loss = self.loss_fn(logits, label)

                if batch_id % 20 == 0:
                    print(f"epoch: {epoch}, batch_id: {batch_id}, loss is: {avg_loss.item():.4f}")
                # 反向传播，更新权重，清除梯度
                self.optimizer.zero_grad()
                avg_loss.backward()
                self.optimizer.step()

            acc = self.evaluate_pm(val_datadir, csv_file)
            if acc > self.best_acc:
                self.save_model(save_path)
                self.best_acc = acc

    # 模型评估阶段
    def evaluate_pm(self, val_datadir, csv_file):
        self.model.eval()
        accuracies = []
        losses = []
        # 验证数据读取器
        valid_loader = valid_data_loader(val_datadir, csv_file)

        with torch.no_grad():
            for img, label in valid_loader:
                img, label = img.to(self.model.device), label.to(self.model.device)
                # 运行模型前向计算，得到预测值
                logits = self.model(img)
                loss = self.loss_fn(logits, label)
                _, predicted = torch.max(logits, 1)
                acc = (predicted == label).sum().item() / label.size(0)
                accuracies.append(acc)
                losses.append(loss.item())

        print(f"[validation] accuracy/loss: {np.mean(accuracies):.4f}/{np.mean(losses):.4f}")
        return np.mean(accuracies)

    # 模型预测阶段
    def predict_pm(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        with torch.no_grad():
            logits = self.model(x)
        return logits

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), os.path.join(save_path, 'palm.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, 'palm_opt.pth'))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

# 示例：自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, datadir, mode='train'):
        # 加载数据逻辑
        self.data = []
        self.labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label

# 示例：自定义验证 Dataset 类
class CustomValidDataset(Dataset):
    def __init__(self, datadir, csv_file):
        # 加载验证数据逻辑
        self.data = []
        self.labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label