#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：Discharge_predict.py
@Author  ：Dai Yikai
@Date    ：2024/3/9 0:11 
@Function：
"""

from utils.data_process import read_from_dataset_folders, add_5days_before, Z_score, min_max
from utils.feature_select import Feature_Select

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Discharge_Predict:
    """
    径流量预测，获取最优模型

    BP神经网络：BPNN_Discharge；
    支持向量机：SVM_Discharge；
    网格搜索交叉验证：Grid_search_CV。

    Attributes:
        data: 输入数据
        features: 选择的特征
        train_size: 训练集比例
        model_name: 模型名称
    """

    def __init__(self, data, features, train_size=0.8, model_name=None):
        self.data = data
        self.features = features
        self.features_names = features.tolist()
        self.train_size = train_size
        self.X_train, self.X_test, self.y_train, self.y_test = self._data_split(self.train_size)
        self.test_MSE = None
        self.model = None
        self.model_name = model_name

    def _data_split(self, train_size=0.8):
        """
        划分训练集和测试集

        :param train_size: 训练集比例
        :return: X_train, X_test, y_train, y_test
        """
        X = data.loc[:, features]
        y = data["Discharge"]
        data_size = len(data)
        train_size = int(data_size * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

    def _Visualization(self, y_train_predict, y_test_predict):
        """
        可视化

        :param y_train_predict: 训练集预测值
        :param y_test_predict: 测试集预测值
        :return: None
        """

        print("-" * 20 + self.model_name + "模型评估" + "-" * 20)
        print("使用特征：", self.features_names)
        print("训练集均方误差：", mean_squared_error(self.y_train, y_train_predict))
        print("训练集R2：", r2_score(self.y_train, y_train_predict))
        print("测试集均方误差：", mean_squared_error(self.y_test, y_test_predict))
        print("测试集R2：", r2_score(self.y_test, y_test_predict))
        print("-" * 50)
        # 可视化
        # 标题：训练集和测试集的真实值与预测值对比
        plt.title("{}模型训练集的真实值与预测值对比".format(self.model_name))
        plt.plot(range(len(self.y_train)), y_train_predict, label="predict")
        plt.plot(range(len(self.y_train)), self.y_train, label="true")
        plt.legend()
        plt.show()
        plt.title("{}训练集的真实值与预测值对比".format(self.model_name))
        plt.plot(range(len(self.y_test)), y_test_predict, label="predict")
        plt.plot(range(len(self.y_test)), self.y_test, label="true")
        plt.legend()
        plt.show()

    def BPNN_Discharge(self, hidden_layer_sizes=(100, 100, 100), learning_rate_init=0.001, activation='relu',
                       solver='adam'):
        """
        BP神经网络预测

        :param hidden_layer_sizes: 隐藏层神经元数目
        :param learning_rate_init: 初始学习率
        :param activation: 激活函数
        :param solver: 优化器

        :return: model
        """

        self.model_name = "BP神经网络"
        # 开始训练
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000,
                             learning_rate_init=learning_rate_init, activation=activation, solver=solver)
        model.fit(self.X_train, self.y_train)
        y_train_predict = model.predict(self.X_train)
        y_test_predict = model.predict(self.X_test)

        self._Visualization(y_train_predict, y_test_predict)

        return mean_squared_error(self.y_test, y_test_predict)

    def SVM_Discharge(self, kernel='linear', C=1.0, gamma='scale', tol=1e-3):
        """
        SVM预测

        :param kernel: 核函数
        :param C: 惩罚系数
        :param gamma: 核系数
        :param tol: 容忍度
        :param is_grid_search: 是否进行网格搜索交叉验证

        :return: 测试集MSE
        """

        self.model_name = "SVM"
        # 开始训练
        model = SVR(kernel=kernel, C=C, gamma=gamma, max_iter=1000, tol=tol)
        model.fit(self.X_train, self.y_train)
        y_train_predict = model.predict(self.X_train)
        y_test_predict = model.predict(self.X_test)

        # 可视化
        self._Visualization(y_train_predict, y_test_predict)

        # 返回测试集MSE
        return mean_squared_error(self.y_test, y_test_predict)

    def Grid_search_CV(self, model_name, cv=3, is_visual=True):
        """
        网格搜索交叉验证

        :param data: 输入数据
        :param features: 选择的特征
        :param train_size: 训练集比例
        :param cv: 交叉验证次数

        """

        # 对BPNN进行网格搜索交叉验证
        if model_name == "BP神经网络":
            self.model_name = "BP神经网络"
            param_grid = {
                'hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), (150, 150, 150)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'activation': ['relu', 'logistic', 'tanh'],
                'solver': ['adam', 'sgd']
            }
            model = GridSearchCV(MLPRegressor(), param_grid, cv=cv, verbose=1)
            model.fit(self.X_train, self.y_train)
            y_train_predict = model.predict(self.X_train)
            y_test_predict = model.predict(self.X_test)

            # 可视化
            print(f"{self.model_name}的最优参数：{model.best_params_}")
            if is_visual:
                self._Visualization(y_train_predict, y_test_predict)
        else:
            self.model_name = "SVM"

            param_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'tol': [1e-3, 1e-4, 1e-5]
            }
            model = GridSearchCV(SVR(), param_grid, cv=cv, verbose=1)
            model.fit(self.X_train, self.y_train)
            y_train_predict = model.predict(self.X_train)
            y_test_predict = model.predict(self.X_test)

            print(f"{self.model_name}的最优参数：{model.best_params_}")
            # 可视化
            if is_visual:
                self._Visualization(y_train_predict, y_test_predict)

        return mean_squared_error(self.y_test, y_test_predict)


if __name__ == "__main__":
    data = read_from_dataset_folders()
    data = add_5days_before(data)
    data = min_max(data)
    selector = Feature_Select()
    features = selector.Pearson_Correlation2nfeature(data, 5)
    discharge_predict = Discharge_Predict(data, features)
    mse_ = discharge_predict.Grid_search_CV("BP神经网络", 5)

