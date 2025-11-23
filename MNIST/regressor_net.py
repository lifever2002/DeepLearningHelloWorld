import torch


class RegressorNet(torch.nn.Module):
    def __init__(self):
        super(RegressorNet, self).__init__()
        self.fc = torch.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs
