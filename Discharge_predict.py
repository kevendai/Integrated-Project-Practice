#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：Discharge_predict.py
@Author  ：Dai Yikai
@Date    ：2024/3/9 0:11 
@Function：
"""

from utils.data_process import read_from_dataset_folders, add_5days_before, Z_score
from utils.feature_select import Feature_Select

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def data_split(data, features, train_size=0.8):
    """
    划分训练集和测试集

    :param data: 输入数据
    :param train_size: 训练集比例
    :return: 训练集和测试集
    """
    X = data.loc[:, features]
    y = data["Discharge"]
    data_size = len(data)
    train_size = int(data_size * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


# 可视化
def Visualization(y_train, y_train_predict, y_test, y_test_predict, model_name, features_names):
    """
    可视化

    :param y_train: 训练集真实值
    :param y_train_predict: 训练集预测值
    :param y_test: 测试集真实值
    :param y_test_predict: 测试集预测值
    """
    print("-" * 20 + model_name + "模型评估" + "-" * 20)
    print("使用特征：", features_names)
    print("训练集均方误差：", mean_squared_error(y_train, y_train_predict))
    print("训练集R2：", r2_score(y_train, y_train_predict))
    print("测试集均方误差：", mean_squared_error(y_test, y_test_predict))
    print("测试集R2：", r2_score(y_test, y_test_predict))
    print("-" * 50)
    # 可视化
    # 标题：训练集和测试集的真实值与预测值对比
    plt.title("{}模型训练集的真实值与预测值对比".format(model_name))
    plt.plot(range(len(y_train)), y_train, label="true")
    plt.plot(range(len(y_train)), y_train_predict, label="predict")
    plt.legend()
    plt.show()
    plt.title("{}训练集的真实值与预测值对比".format(model_name))
    plt.plot(range(len(y_test)), y_test, label="true")
    plt.plot(range(len(y_test)), y_test_predict, label="predict")
    plt.legend()
    plt.show()


def BPNN_Discharge(data, features, train_size=0.8, hidden_layer_sizes=(100, 100, 100), max_iter=1000):
    """
    BP神经网络预测

    :param data: 输入数据
    :param features: 选择的特征
    :param train_size: 训练集比例
    :param hidden_layer_sizes: 隐藏层神经元数目
    :param max_iter: 最大迭代次数
    """

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = data_split(data, features, train_size)

    # 开始训练
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    # 可视化
    Visualization(y_train, y_train_predict, y_test, y_test_predict, "BP神经网络", features.tolist())

    return model


def SVM_Discharge(data, features, train_size=0.8):
    """
    SVM预测

    :param data: 输入数据
    :param features: 选择的特征
    :param train_size: 训练集比例
    """

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = data_split(data, features, train_size)

    # 开始训练
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    # 可视化
    Visualization(y_train, y_train_predict, y_test, y_test_predict, "SVM", features.tolist())

    return model


if __name__ == "__main__":
    data = read_from_dataset_folders()
    data = add_5days_before(data)
    data = Z_score(data)
    selector = Feature_Select()
    features = selector.Pearson_Correlation2nfeature(data, 5)
    model = SVM_Discharge(data, features)
