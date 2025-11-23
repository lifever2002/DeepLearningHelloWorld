import torch
from load_data import load_data
from regressor import Regressor

ten, twenty, thirty = 0, 0, 0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Regressor()
# 加载模型参数
model_dict = torch.load("bh_torch.pt")
model.load_state_dict(model_dict)
# 开启模型评估模式
model.eval()
# CUDA
model.to(DEVICE)
print("Model has been moved to {}".format(DEVICE))
training_data, test_data, max_values, min_values = load_data("bh.csv")
hundred = len(test_data)

for i, one_data in enumerate(test_data):
    x, label = one_data[:-1], one_data[-1]
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
        predict = model(x)
    predict = predict.item() * (max_values[-1] - min_values[-1]) + min_values[-1]
    label = label * (max_values[-1] - min_values[-1]) + min_values[-1]

    print(f"label: {label}, predict: {predict}, error: {predict-label}")

    error = abs((predict - label) / label)
    if error < 0.1:
        ten += 1
    if error < 0.2:
        twenty += 1
    if error < 0.3:
        thirty += 1

print(f"error under 10% : {ten/hundred*100:.2f}% ({ten} of {hundred})")
print(f"error under 20% : {twenty/hundred*100:.2f}% ({twenty} of {hundred})")
print(f"error under 30% : {thirty/hundred*100:.2f}% ({thirty} of {hundred})")
