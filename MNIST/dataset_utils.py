import torch
from torchvision import datasets, transforms


def load_data(train=True):
    tf = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST(root="data", train=train, transform=tf, download=True)
    return data


def norm_data(data):
    if len(data.shape) == 4:  # 如果是 [batch_size, 1, 28, 28]
        data = data.squeeze(1)  # 去掉通道维度，变为 [batch_size, 28, 28]
    batch_size, height, width = data.shape[0], data.shape[1], data.shape[2]
    # 归一化
    data = data / 255
    data = torch.reshape(data, [batch_size, height * width])
    return data
