import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 切换工作目录
base_dir = os.path.dirname(__file__)
os.chdir(base_dir)

# 读取数据
lines = np.loadtxt("lr_dataset.csv", delimiter=",", dtype=np.float32)
X = lines[:, :-1]
y = lines[:, -1]

# 打乱数据
np.random.seed(0)
index = np.random.permutation(X.shape[0])
X = X[index]
y = y[index]

# 划分训练集和测试集
ratio = 0.7
split_index = int(X.shape[0] * ratio)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# 添加偏置项
X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=-1)
X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=-1)

# 设置超参数
epochs = 250
learning_rate = 0.002
l2_coef = 1.0


# y_pred 为类别 0, 1
def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)


# y_pred 为预测概率 (0, 1)
def auc(y_true, y_pred):
    # y_true 按 y_pred 降序排序
    index = np.argsort(y_pred)[::-1]
    y_true = y_true[index]
    y_pred = y_pred[index]

    # 通过累加（cumsum）计算不同阈值（y_pred）下的TP和FP
    tp = np.cumsum(y_true)  # tp[i]表示阈值为y_pred[i]时的TP数量
    fp = np.cumsum(1 - y_true)  # fp[i]表示阈值为y_pred[i]时的FP数量

    # 阈值最低时，TP和FP的数量 = 正样本数和负样本数
    P = tp[-1]
    N = fp[-1]

    # 计算tpr和fpr
    tpr = tp / P
    fpr = fp / N

    # 计算AUC（使用矩形法则）
    # x轴是fpr，y轴是tpr
    s = 0.0
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    for i in range(1, len(tpr)):
        s += (fpr[i] - fpr[i - 1]) * tpr[i]
    return s


def logistic(z):
    return 1 / (1 + np.exp(-z))


def GD(epochs, learning_rate, l2_coef):
    theta = np.random.normal(size=(X_train.shape[1],))
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    train_auc, test_auc = [], []

    for i in range(epochs):
        # 初始化参数
        pred = logistic(X_train @ theta)
        grad = -X_train.T @ (y_train - pred) + l2_coef * theta
        theta -= learning_rate * grad

        # 记录损失函数
        train_loss = (
            -y_train.T @ np.log(pred)
            - (1 - y_train).T @ np.log(1 - pred)
            + l2_coef / 2 * np.linalg.norm(theta) ** 2
        )
        train_loss /= X_train.shape[0]
        train_losses.append(train_loss)

        test_pred = logistic(X_test @ theta)
        test_loss = -y_test.T @ np.log(test_pred) - (1 - y_test).T @ np.log(
            1 - test_pred
        )
        test_loss /= y_test.shape[0]
        test_losses.append(test_loss)

        # 记录 acc
        train_acc.append(acc(y_train, pred >= 0.5))
        test_acc.append(acc(y_test, test_pred >= 0.5))

        # 记录auc
        train_auc.append(auc(y_train, pred))
        test_auc.append(auc(y_test, test_pred))

    return theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc


theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc = GD(
    epochs, learning_rate, l2_coef
)

# 测试模型准确率
pred = logistic(X_test @ theta)
acc = acc(y_test, pred >= 0.5)
print(f"回归系数：{theta}")
print(f"准确率：{acc}")

plt.figure(figsize=(13, 9))
xticks = np.arange(epochs) + 1

# 绘制训练曲线
plt.subplot(221)
plt.plot(xticks, train_losses, color="b", label="train loss")
plt.plot(xticks, test_losses, color="r", ls="--", label="test loss")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()

# 绘制准确率
plt.subplot(222)
plt.plot(xticks, train_acc, color="b", label="train accuracy")
plt.plot(xticks, test_acc, color="r", ls="--", label="test accuracy")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoches")
plt.ylabel("Accuracy")
plt.legend()

# 绘制AUC
plt.subplot(223)
plt.plot(xticks, train_auc, color="b", label="train AUC")
plt.plot(xticks, test_auc, color="r", ls="--", label="test AUC")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoches")
plt.ylabel("AUC")
plt.legend()

# 绘制决策边界
plt.subplot(224)
# z = X @ theta = theta_0 * x_1 + theta_1 * x_2 + theta_2
# 当σ(z) >= 0.5 时，z >= 0
# x_2 = -(theta_0 * x_1 + theta_2) / theta_1
x1 = np.linspace(-1.1, 1.1, 100)
x2 = -(theta[0] * x1 + theta[2]) / theta[1]
plt.plot(x1, x2, ls="-.", color="g")

pos_index = np.where(y == 1)
neg_index = np.where(y == 0)
plt.scatter(X[pos_index, 0], X[pos_index, 1], marker="o", color="r", s=10)
plt.scatter(X[neg_index, 0], X[neg_index, 1], marker="x", color="b", s=10)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
