import os

import numpy as np


def classify_0(features, labels, x, k):
    distances_2 = np.square(features - x)
    distances = np.sqrt(np.sum(distances_2, axis=1))

    sorted_index = np.argsort(distances)

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    return max(class_count, key=class_count.get)


def file2matrix(filename):
    data = np.loadtxt(filename, delimiter="\t")
    features = data[:, :-1]
    labels = data[:, -1].astype(int)

    return features, labels


def auto_norm(features):
    min_vals = features.min(0)
    max_vals = features.max(0)
    ranges = max_vals - min_vals
    norm_features = (features - min_vals) / ranges

    return norm_features


def dating_train():
    base_dir = os.path.dirname(__file__)
    os.chdir(base_dir)

    filename = "datingTestSet2.txt"
    features, labels = file2matrix(filename)

    ratio = 0.7
    features = auto_norm(features)
    split_index = int(features.shape[0] * ratio)
    train_features = features[:split_index]
    train_labels = labels[:split_index]

    k = 3
    errcount = 0
    for i in range(train_features.shape[0]):
        x = train_features[i]
        pred_label = classify_0(train_features, train_labels, x, k)
        if pred_label != train_labels[i]:
            errcount += 1

    print(f"Error rate: {errcount / (train_features.shape[0]):.2f}")
    print(f"错误样本数: {errcount}")


if __name__ == "__main__":
    dating_train()
