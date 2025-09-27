import numpy as np


# 分批数据生成器
def batch_generator(X, Y, batch_size, shuffle=True):
    batch_count = 0

    # 打乱数据
    if shuffle:
        index = np.random.permutation(X.shape[0])
        X = X[index]
        Y = Y[index]

    # 分批生成数据
    while True:
        start = batch_count * batch_size
        end = min(start + batch_size, X.shape[0])  # 防止索引越界
        if start >= end:
            break
        yield X[start:end], Y[start:end]
        batch_count += 1


def SGD(X_train, Y_train, X_test, Y_test, num_epochs, batch_size, learning_rate):
    # 数据预处理
    # 添加偏置项
    X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=-1)
    X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=-1)
    theta = np.random.normal(size=X_train.shape[1])  # 初始化回归系数
    test_losses = []

    # 迭代训练
    for epoch in range(num_epochs):
        # 生成分批数据
        batch_g = batch_generator(X_train, Y_train, batch_size)

        # 每批数据进行梯度下降
        for X_batch, Y_batch in batch_g:
            grad = (X_batch.T @ (X_batch @ theta - Y_batch)) / batch_size
            theta -= learning_rate * grad

        # 计算测试集损失（MSE）
        test_loss = (
            ((X_test @ theta - Y_test).T @ (X_test @ theta - Y_test)) / 2
        ) / X_test.shape[0]
        test_losses.append(test_loss)

    print("回归系数：", theta)
    return theta, test_losses
