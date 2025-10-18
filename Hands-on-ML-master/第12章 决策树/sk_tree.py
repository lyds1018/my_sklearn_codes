import os

import numpy as np
import pandas as pd
import pydotplus
from six import StringIO
from sklearn import preprocessing, tree

# 一、读取数据
base_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir, "titanic\\train.csv"))
data.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)  # 删除无关特征


# 二、缺失值处理与特征编码
# 定义特征类型
continuous_or_count_feats = ["Age", "Fare", "SibSp", "Parch"]  # 数值型
ordinal_categorical_feats = ["Pclass"]  # 有序类别
nominal_categorical_feats = ["Sex", "Cabin", "Embarked"]  # 无序类别

# 处理数值型和有序类别特征
# 这些特征可以直接被决策树使用，只需填充缺失值
numerical_feats = continuous_or_count_feats + ordinal_categorical_feats
data[numerical_feats] = data[numerical_feats].fillna(data[numerical_feats].median())

# 处理无序类别特征
# 需要填充缺失值并进行整数编码
data[nominal_categorical_feats] = data[nominal_categorical_feats].fillna("missing")
encoder = preprocessing.OrdinalEncoder()
data[nominal_categorical_feats] = encoder.fit_transform(data[nominal_categorical_feats])


# 三、划分训练集和测试集
np.random.seed(0)
label_name = "Survived"
feat_names = data.columns.drop(label_name)

# 随机打乱数据
data = data.sample(frac=1).reset_index(drop=True)

# 划分特征和标签
X = data[feat_names].to_numpy()
y = data[label_name].to_numpy()

# 按比例分割
ratio = 0.8
split_idx = int(ratio * len(data))
train_x, test_x = X[:split_idx], X[split_idx:]
train_y, test_y = y[:split_idx], y[split_idx:]


# 四、构建、训练与评估模型
# 构建模型
c45 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6)  # C4.5
cart = tree.DecisionTreeClassifier(criterion="gini", max_depth=6)  # CART

# 训练模型
c45.fit(train_x, train_y)
cart.fit(train_x, train_y)


# 评估模型
def accuracy(pred, true):
    return np.mean(pred == true)


c45_train_acc = accuracy(c45.predict(train_x), train_y)
c45_test_acc = accuracy(c45.predict(test_x), test_y)
cart_train_acc = accuracy(cart.predict(train_x), train_y)
cart_test_acc = accuracy(cart.predict(test_x), test_y)

print(f"训练集准确率：C4.5 = {c45_train_acc:.4f}，CART = {cart_train_acc:.4f}")
print(f"测试集准确率：C4.5 = {c45_test_acc:.4f}，CART = {cart_test_acc:.4f}")


# 五、可视化决策树（C4.5）
dot_data = StringIO()
tree.export_graphviz(
    c45,
    out_file=dot_data,  # 决策树数据导入到 dot_data
    feature_names=feat_names,  # 特征名称
    class_names=["non-survival", "survival"],  # 类别名称
)

# 使用 pydotplus 将 dot_data 转换为图像
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", ""))
graph.write_png("tree.png")

print("\n决策树已导出为 tree.png")
