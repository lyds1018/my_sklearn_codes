import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 加载数据
base_dir = os.path.dirname(__file__)  # 获取当前文件所在目录
os.chdir(base_dir)  # 切换工作目录
lines = np.loadtxt("USA_Housing.csv", delimiter=",", dtype=str)
lines = lines[1:].astype(np.float32)


# 数据预处理
# 数据集划分
ratio = 0.8
split_index = int(lines.shape[0] * ratio)
lines = np.random.permutation(lines)  # 打乱数据集
train, test = lines[:split_index], lines[split_index:]  # 划分训练集和测试集

# 数据标准化 (标准化公式: z = (x - u) / s )
scaler = StandardScaler()  # 创建一个标准化器对象
scaler.fit(train)  # 使用训练集的均值 u 和 方差 s^2, 初始化标准化器
train = scaler.transform(train)
test = scaler.transform(test)

# 划分特征与标签
x_train, y_train = train[:, :-1], train[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]


# 线性回归
# theta = (X^T * X)^-1 * X^T * y
X = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=-1)
theta = np.linalg.inv(X.T @ X) @ X.T @ y_train
print("Theta:", theta)

# 预测
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=-1)
y_pred = X_test @ theta

# 评估
mse = np.mean((y_pred - y_test) ** 2)
print("MSE:", mse)


# 使用 sklearn 内置模型进行线性回归
linreg = LinearRegression()  # 初始化线性回归模型
linreg.fit(x_train, y_train)  # 内置模型自动添加偏置项
y_pred_sklearn = linreg.predict(x_test)
mse_sklearn = np.mean((y_pred_sklearn - y_test) ** 2)
print("Theta:", np.concatenate([linreg.coef_, [linreg.intercept_]], axis=0))
print("MSE_sklearn:", mse_sklearn)
