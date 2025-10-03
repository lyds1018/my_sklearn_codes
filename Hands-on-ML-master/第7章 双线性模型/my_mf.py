import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 进度条工具


class MF:
    def __init__(self, N, M, d):
        self.P = np.ones((N, d))  # 用户特征矩阵
        self.Q = np.ones((M, d))  # 物品特征矩阵

    def predict(self, i, j):
        return np.sum(self.P[i] * self.Q[j], axis=1)

    def update(self, grad_P, grad_Q, lr):
        self.P -= lr * grad_P
        self.Q -= lr * grad_Q


def train_mf(
    model,
    lr,
    lbd,
    epochs,
    batch_size,
    user_train,
    item_train,
    y_train,
    user_test,
    item_test,
    y_test,
):
    train_losses = []
    test_losses = []
    batch_num = int(np.ceil(len(user_train) / batch_size))
    with tqdm(range(epochs * batch_num)) as pbar:
        for epoch in range(epochs):
            train_rmse = 0
            for batch_idx in range(batch_num):
                st = batch_idx * batch_size
                ed = min(len(user_train), st + batch_size)
                user_batch = user_train[st:ed]
                item_batch = item_train[st:ed]
                y_batch = y_train[st:ed]

                y_pred = model.predict(user_batch, item_batch)
                errs = y_batch - y_pred

                grad_P = np.zeros(model.P.shape)
                grad_Q = np.zeros(model.Q.shape)
                for u_id, i_id, err in zip(user_batch, item_batch, errs):
                    grad_P[u_id] += -err * model.Q[i_id] + lbd * model.P[u_id]
                    grad_Q[i_id] += -err * model.P[u_id] + lbd * model.Q[i_id]
                grad_P = grad_P / len(user_batch)
                grad_Q = grad_Q / len(user_batch)

                model.update(grad_P, grad_Q, lr)

                train_rmse += np.mean(errs**2)

                # 更新进度条
                pbar.set_postfix(
                    {
                        "Epoch": epoch,
                        "Train RMSE": f"{np.sqrt(train_rmse / (batch_idx + 1)):.4f}",
                        "Test RMSE": f"{test_losses[-1]:.4f}" if test_losses else None,
                    }
                )
                pbar.update(1)

            # 计算 RMSE 损失
            train_rmse = np.sqrt(train_rmse / batch_num)
            train_losses.append(train_rmse)
            y_test_pred = model.predict(user_test, item_test)
            test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            test_losses.append(test_rmse)

    return train_losses, test_losses


# 超参数
feature_num = 16  # 特征数
learning_rate = 0.1  # 学习率
lbd = 1e-4  # 正则化强度
epochs = 30
batch_size = 64  # 批量大小

# 读取数据
base_dir = os.path.dirname(__file__)
os.chdir(base_dir)
data = np.loadtxt("movielens_100k.csv", delimiter=",", dtype=int)
data[:, :2] = data[:, :2] - 1  # 用户ID和电影ID从0开始编号

users = set()
items = set()
for u, i, r in data:
    users.add(u)
    items.add(i)
N = len(users)  # 用户数
M = len(items)  # 物品数

# 设置随机种子，划分训练集与测试集
np.random.seed(0)
ratio = 0.8
split = int(len(data) * ratio)
np.random.shuffle(data)
train = data[:split]
test = data[split:]

# 划分输入输出
user_train, user_test = train[:, 0], test[:, 0]
item_train, item_test = train[:, 1], test[:, 1]
y_train, y_test = train[:, 2], test[:, 2]

# 建立模型
model = MF(N, M, feature_num)
# 训练部分
train_losses, test_losses = train_mf(
    model,
    learning_rate,
    lbd,
    epochs,
    batch_size,
    user_train,
    item_train,
    y_train,
    user_test,
    item_test,
    y_test,
)

plt.figure()
x = np.arange(epochs) + 1
plt.plot(x, train_losses, color="blue", label="train loss")
plt.plot(x, test_losses, color="red", ls="--", label="test loss")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()
