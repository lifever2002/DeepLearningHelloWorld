import torch
import torch.optim.adagrad
import torch.optim.adam
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from dataset_utils import load_data, norm_data
from regressor_net import RegressorNet
from fc_net import FCNet
from cnn import CNN
from torch.utils.tensorboard import SummaryWriter


# parameters
MODEL_PATH = "checkpoints/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 1024
LR = 1e-2
PRINT_EVERY = 1
criterion = torch.nn.CrossEntropyLoss()

eval_losses = []
model = CNN()
# 开启模型训练模式
model.train()
# CUDA
model.to(DEVICE)
print("Model has been moved to {}".format(DEVICE))


train_loader = DataLoader(load_data(train=True), batch_size=BATCH_SIZE, shuffle=True)
# opt = torch.optim.SGD(lr=LR, params=model.parameters())
# opt = torch.optim.Adagrad(lr=LR, params=model.parameters())
log_writer = SummaryWriter(log_dir="log")


# 定义总步数
total_steps = (50000 // BATCH_SIZE + 1) * EPOCHS
opt = torch.optim.Adam(lr=LR, weight_decay=1e-5, params=model.parameters())


# 定义学习率调度器
lr_scheduler = PolynomialLR(optimizer=opt, total_iters=total_steps, power=1.0)

for epoch in range(EPOCHS):
    for i, data in enumerate(train_loader):
        images = data[0].to(DEVICE, dtype=torch.float32)  # CNN
        # images = norm_data(data[0]).to(DEVICE, dtype=torch.float32)
        labels = data[1].to(DEVICE, dtype=torch.int64)

        predicts = model(images)

        accuracy = Accuracy(task="multiclass", num_classes=10).to(DEVICE)
        acc = accuracy(predicts, labels)

        avg_loss = criterion(predicts, labels)
        eval_losses.append(avg_loss.item())
        if epoch % PRINT_EVERY == 0:
            print(
                f"epoch: {epoch}, loss: {avg_loss.item():.4f}, accuracy: {acc.item()*100:.2f}%"
            )
            log_writer.add_scalar(tag="acc", global_step=epoch, scalar_value=acc.item())
            log_writer.add_scalar(
                tag="loss", global_step=epoch, scalar_value=avg_loss.item()
            )

        avg_loss.backward()  # 计算梯度
        opt.step()  # 更新参数
        lr_scheduler.step()
        opt.zero_grad()  # 清空梯度

    torch.save(
        model.state_dict(), "{}mnist_epoch_{}".format(MODEL_PATH, epoch) + ".pth"
    )
    torch.save(opt.state_dict(), "{}mnist_epoch_{}".format(MODEL_PATH, epoch) + ".opt")

torch.save(model.state_dict(), "mnist.pt")
plt.plot(np.arange(len(eval_losses)), np.array(eval_losses))
plt.show()
