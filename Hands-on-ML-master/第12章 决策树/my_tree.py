import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = float(dataSet.shape[0])

    # 使用 np.unique 计算每种类别的数量
    _, labelCounts = np.unique(dataSet[:, -1], return_counts=True)

    probs = labelCounts / numEntries
    shannonEnt = -np.sum(probs * np.log2(probs))
    return shannonEnt


# 2. 按给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    # 给定特征对应第 axis 列
    # 返回第 axis 列上值为 value 的行
    mask = dataSet[:, axis] == value
    # 删除用于划分的第 axis 列
    retDataSet = np.delete(dataSet[mask], axis, axis=1)
    return retDataSet


# 3. 选择最优划分特征（信息增益最大）
def chooseBestFeatureToSplit(dataSet):
    numFeatures = dataSet.shape[1] - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 划分前的香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获取第 i 列的所有特征值
        featList = dataSet[:, i]
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        # 对所有类别加权求系统熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = subDataSet.shape[0] / float(dataSet.shape[0])
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 4. 多数投票
def majorityCnt(classList):
    # 获取唯一的类别及其计数
    vals, counts = np.unique(classList, return_counts=True)
    # 找到计数最大值的索引
    index = np.argmax(counts)
    # 返回计数最多的类别
    return vals[index]


# 5. 递归构建决策树
def createTree(dataSet, labels):
    # 两个递归停止条件
    classList = dataSet[:, -1]
    # 如果所有类别都相同，则停止
    if np.unique(classList).size == 1:
        return classList[0]
    # 如果所有特征都用完了，返回多数投票结果
    if dataSet.shape[1] == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)  # 当前最优划分特征索引
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 字典存储当前树节点

    # 内层递归的特征标签列表
    subLabels = labels[:]
    del subLabels[bestFeat]  # 删除已使用的特征

    # 根据当前划分特征的所有类别，递归调用 createTree
    featValues = dataSet[:, bestFeat]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 内层递归返回的子树节点，嵌套在当前树节点中
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels
        )

    # 返回当前树节点
    return myTree


# 6. 分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]  # 获取子树
    featIndex = featLabels.index(firstStr)  # 找到样本当前特征对应的索引
    for key in secondDict.keys():
        # 匹配样本的当前特征值
        if testVec[featIndex] == key:
            # 判断子树下一层是叶节点还是树节点
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 7. 保存树
def storeTree(inputTree, filename):
    with open(filename, "wb") as fw:
        pickle.dump(inputTree, fw)


# 8. 读取树
def grabTree(filename):
    with open(filename, "rb") as fr:
        return pickle.load(fr)


# ---------------- 绘图部分 ----------------
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算叶节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt,
        xy=parentPt,
        xycoords="axes fraction",
        xytext=centerPt,
        textcoords="axes fraction",
        va="center",
        ha="center",
        bbox=nodeType,
        arrowprops=arrow_args,
    )


# 绘制中间文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 绘制树
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (
        plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
        plotTree.yOff,
    )
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 创建绘图
def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), "")
    plt.show()


# ---------------- 主程序部分 ----------------
def main():
    # 读取数据
    basedir = os.path.dirname(__file__)
    os.chdir(basedir)
    fr = pd.read_csv("lenses.txt", sep="\t", header=None)
    # 直接使用 fr.values 获取 NumPy 数组，不再转换为 list
    lenses = fr.values
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]

    # 创建决策树
    lensesTree = createTree(lenses, lensesLabels[:])

    # 测试分类
    testVec = ["young", "myope", "no", "normal"]
    result = classify(lensesTree, lensesLabels, testVec)
    print(f"测试样本 {testVec} 的预测结果为：{result}")

    # 绘制
    createPlot(lensesTree)

    # 保存
    storeTree(lensesTree, "lenses_tree.pkl")


if __name__ == "__main__":
    main()
