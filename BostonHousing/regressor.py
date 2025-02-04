import torch


class Regressor(torch.nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = torch.nn.Linear(in_features=13, out_features=1)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x
