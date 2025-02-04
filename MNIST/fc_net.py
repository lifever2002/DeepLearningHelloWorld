import torch
from torch.nn import Linear


# 定义多层全连接神经网络
class FCNet(torch.nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # 定义两层全连接隐含层，输出维度是10，当前设定隐含节点数为10，可根据任务调整
        self.fc1 = Linear(in_features=784, out_features=10)
        self.fc2 = Linear(in_features=10, out_features=10)
        # 定义一层全连接输出层，输出维度是1
        self.fc3 = Linear(in_features=10, out_features=1)

    # 定义网络的前向计算，隐含层激活函数为sigmoid，输出层不使用激活函数
    def forward(self, inputs):
        inputs = torch.reshape(inputs, [inputs.shape[0], 784])
        outputs1 = self.fc1(inputs)
        outputs1 = torch.sigmoid(outputs1)
        outputs2 = self.fc2(outputs1)
        outputs2 = torch.sigmoid(outputs2)
        outputs_final = self.fc3(outputs2)
        return outputs_final
