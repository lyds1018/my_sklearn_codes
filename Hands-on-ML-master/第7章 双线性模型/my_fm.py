import os

import numpy as np
from sklearn import metrics  # sklearn中的评价指标函数库
from tqdm import tqdm


class FM:
    def __init__(self, feature_num, vector_dim):
        self.theta0 = 0
        self.theta = np.zeros((feature_num, 1))
        self.V = np.random.normal(size=(feature_num, vector_dim))
        self.eps = 1e-8  # 防止除0错误

    def logistic(self, z):
        return 1.0 / (1 + np.exp(-z))

    def predict(self, X):
        # 线性部分
        linear_term = self.theta0 + X @ self.theta
        # 双线性部分
        square_of_sum = np.square(X @ self.V)
        sum_of_square = np.square(X) @ np.square(self.V)
        # Z
        Z = linear_term + 0.5 * np.sum(square_of_sum - sum_of_square, axis=1).reshape(
            -1, 1
        )  # axis=1, 每行按列求和
        # 最终预测
        Y_pred = self.logistic(
            Z
        )  # 为了防止后续梯度过大，对预测值进行裁剪，将其限制在某一范围内
        Y_pred = np.clip(
            Y_pred, self.eps, 1 - self.eps
        )  # np.clip(a, low, high) 会把数组 a 中的值截断到区间 [low, high]
        return Y_pred

    def update(self, grad0, grad_theta, grad_V, lr):
        self.theta0 -= lr * grad0
        self.theta -= lr * grad_theta
        self.V -= lr * grad_V


# 设置工作目录
base_dir = os.path.dirname(__file__)
os.chdir(base_dir)

# 导入数据集
data = np.loadtxt("fm_dataset.csv", delimiter=",")

# 划分数据集
np.random.seed(0)
ratio = 0.8
split = int(ratio * len(data))
X_train = data[:split, :-1]
Y_train = data[:split, -1].reshape(-1, 1)
X_test = data[split:, :-1]
Y_test = data[split:, -1].reshape(-1, 1)

# 特征数
feature_num = X_train.shape[1]

# 超参数设置，包括学习率、训练轮数等
vector_dim = 16
learning_rate = 0.01
lbd = 0.05
max_training_step = 200
batch_size = 32

# 初始化模型
np.random.seed(0)
model = FM(feature_num, vector_dim)

train_acc = []
test_acc = []
train_auc = []
test_auc = []

# 训练
with tqdm(range(max_training_step)) as pbar:
    for epoch in pbar:
        st = 0
        while st < X_train.shape[0]:
            ed = min(st + batch_size, X_train.shape[0])
            X_batch = X_train[st:ed]
            Y_batch = Y_train[st:ed]
            st += batch_size

            # 预测
            Y_pred = model.predict(X_batch)
            # 计算损失
            losses = -Y_batch * np.log(Y_pred) - (1 - Y_batch) * np.log(1 - Y_pred)
            loss = np.sum(losses)

            # 计算梯度
            grad_Z = (Y_pred - Y_batch).reshape(-1, 1)
            # 常数项: ∂z_i/∂theta0 = 1
            grad0 = (np.ones((X_batch.shape[0], 1)).T @ grad_Z).item() / X_batch.shape[
                0
            ] + lbd * model.theta0
            # 线性项: ∂Xtheta/∂theta.T = X
            grad_theta = (X_batch.T @ grad_Z) / X_batch.shape[0] + lbd * model.theta
            # 双线性项
            grad_V = np.zeros((feature_num, vector_dim))
            # 对每个样本计算梯度，然后累加
            for k, x in enumerate(X_batch):
                grad_vk = np.zeros((feature_num, vector_dim))

                # 对每个特征计算梯度，然后累加
                xv = x @ model.V
                for s in range(feature_num):
                    grad_vk_s = x[s] * xv - (x[s] ** 2) * model.V[s]
                    grad_vk[s] += grad_vk_s.T * grad_Z[k]
                grad_V += grad_vk

            grad_V = grad_V / X_batch.shape[0] + lbd * model.V

            # 更新参数
            model.update(grad0, grad_theta, grad_V, learning_rate)

            # 进度条显示
            pbar.set_postfix(
                {
                    "训练轮数": epoch + 1,
                    "训练损失": f"{loss:.4f}",
                    "训练集准确率": train_acc[-1] if train_acc else None,
                    "测试集准确率": test_acc[-1] if test_acc else None,
                }
            )

        # 计算在训练集和测试集上的准确率和AUC
        # 预测准确率，阈值设置为0.5
        Y_train_pred = model.predict(X_train) >= 0.5
        acc = np.mean(Y_train_pred == Y_train)
        train_acc.append(acc)
        auc = metrics.roc_auc_score(Y_train, Y_train_pred)  # sklearn中的AUC函数
        train_auc.append(auc)

        Y_test_pred = model.predict(X_test) >= 0.5
        acc = np.mean(Y_test_pred == Y_test)
        test_acc.append(acc)
        auc = metrics.roc_auc_score(Y_test, Y_test_pred)
        test_auc.append(auc)

    print(f"测试集准确率：{test_acc[-1]}，\t测试集AUC：{test_auc[-1]}")
