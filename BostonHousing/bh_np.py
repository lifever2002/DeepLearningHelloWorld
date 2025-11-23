import numpy as np
import matplotlib.pyplot as plt


# 网络->根据逻辑构建
class bhnet(object):
    def __init__(self, w_count):

        # 固定随机数种子、保持运行一致性
        np.random.seed(0)
        # 初始化、w为一维列向量
        self.w = np.random.randn(w_count, 1)
        self.b = 0.0

    # 激活函数->变换计算结果
    def activation(self, y):
        return y

    # 前向传播
    def forward(self, x):
        # 运算
        y = np.dot(x, self.w) + self.b
        # 激活
        z = self.activation(y)
        return z

    # 损失函数->衡量结果好坏
    def loss(self, z, y):
        # 运算值和真实值
        error = z - y
        cost = error * error
        # 取平均、除去样本数量影响
        cost = np.mean(cost)
        return cost  # 数字

    # 求梯度->决定下降的方向
    def gradient(self, x, y):
        z = self.forward(x)
        # x是一维行向量、w是一维列向量
        gradient_w = (z - y) * x  # 由损失函数对w偏微分得来、最低点梯度为0
        gradient_w = np.mean(gradient_w, axis=0)  # Shape=(13)
        gradient_w = gradient_w[:, np.newaxis]  # Shape->(13,1)、行变列
        gradient_b = z - y
        gradient_b = np.mean(gradient_b, axis=0)
        return gradient_w, gradient_b

    # 更新梯度
    def update(self, gradient_w, gradient_b, eta=0.01):
        # eta（学习率）->决定下降的步长
        self.w = self.w - gradient_w * eta
        self.b = self.b - gradient_b * eta

    # 训练
    def train(self, train_data, epochs=10, batch_size=10, eta=0.01):
        losses = []
        for epoch_i in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k : k + batch_size]
                for k in range(0, len(train_data), batch_size)
            ]
            for batch_i, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                z = self.forward(x)
                l = self.loss(z, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(l)
        return losses


def load_data(data_file):
    # 首行（列标题）
    feature_names = np.genfromtxt(data_file, delimiter=",", max_rows=1, dtype=str)
    # 二维数据
    data = np.genfromtxt(data_file, delimiter=",", skip_header=1)

    # 归一化
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    # 划分训练集
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    train_data = data[:offset]
    test_data = data[offset:]

    return train_data, test_data


# 获取数据
train_data, test_data = load_data("bh.csv")

# 创建网络
net = bhnet(13)
epoch = 50
batch_size = 100
losses = net.train(train_data, epoch, batch_size, eta=0.01)

plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
