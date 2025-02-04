import torch
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from regressor import Regressor

# parameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS = 1000
BATCH_SIZE = 110
LR = 5e-3
PRINT_EVERY = 10

eval_losses = []
model = Regressor()
# 开启模型训练模式
model.train()
# CUDA
model.to(DEVICE)
print("Model has been moved to {}".format(DEVICE))
training_data, test_data, max_values, min_values = load_data("bh.csv")
opt = torch.optim.SGD(lr=LR, params=model.parameters())
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    np.random.shuffle(training_data)
    mini_batches = [
        training_data[k : k + BATCH_SIZE]
        for k in range(0, len(training_data), BATCH_SIZE)
    ]

    for i, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])
        y = np.array(mini_batch[:, -1:])
        house_features = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        prices = torch.tensor(y, dtype=torch.float32).to(DEVICE)

        predicts = model(house_features)
        avg_loss = criterion(predicts, prices)
        eval_losses.append(avg_loss.item())
        if (epoch % PRINT_EVERY == 0) & (i == 0):
            print(f"epoch: {epoch}, loss: {avg_loss.item()}")

        avg_loss.backward()  # 计算梯度
        opt.step()  # 更新参数
        opt.zero_grad()  # 清空梯度

torch.save(model.state_dict(), "boston_housing.pt")
plt.plot(np.arange(len(eval_losses)), np.array(eval_losses))
plt.show()
