#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：Discharge_predict.py
@Author  ：Dai Yikai
@Date    ：2024/3/9 0:11 
@Function：
"""
import os
import json
import pandas as pd

from utils.data_process import read_from_dataset_folders, add_5days_before, Z_score, min_max
from utils.data_process import reverse_Z_score, reverse_min_max
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

    def __init__(self, data, features,  reverse_method=None, reverse_param=None, train_size=0.8, model_name=None):
        self.data = data
        self.features = features
        self.features_names = features.tolist()
        self.train_size = train_size
        self.X_train, self.X_test, self.y_train, self.y_test = self._data_split(self.train_size)
        self.test_MSE = None
        self.model = None
        self.model_name = model_name
        if reverse_param is None:
            self.reverse_param1, self.reverse_param2 = None, None
        else:
            self.reverse_param1, self.reverse_param2 = reverse_param
        self.reverse_method = reverse_method

    def _data_split(self, train_size=0.8):
        """
        划分训练集和测试集

        :param train_size: 训练集比例
        :return: X_train, X_test, y_train, y_test
        """
        X = self.data.loc[:, self.features]
        y = self.data["Discharge"]
        data_size = len(self.data)
        train_size = int(data_size * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

    def _Visualization(self, y_train_predict, y_test_predict, is_save=False):
        """
        可视化

        :param y_train_predict: 训练集预测值
        :param y_test_predict: 测试集预测值
        :return: None
        """
        if self.reverse_param1 is None:
            y_train_predict_reverse = y_train_predict
            y_test_predict_reverse = y_test_predict
            y_train_reverse = self.y_train
            y_test_reverse = self.y_test
        else:
            y_train_predict_reverse = self.reverse_method(y_train_predict, self.reverse_param1, self.reverse_param2)
            y_test_predict_reverse = self.reverse_method(y_test_predict, self.reverse_param1, self.reverse_param2)
            y_train_reverse = self.reverse_method(self.y_train, self.reverse_param1, self.reverse_param2)
            y_test_reverse = self.reverse_method(self.y_test, self.reverse_param1, self.reverse_param2)

        print("-" * 20 + self.model_name + "模型评估" + "-" * 20)
        print("使用特征：", self.features_names)
        print("训练集均方误差：", mean_squared_error(y_train_reverse, y_train_predict_reverse))
        print("训练集R2：", r2_score(y_train_reverse, y_train_predict_reverse))
        print("测试集均方误差：", mean_squared_error(y_test_reverse, y_test_predict_reverse))
        print("测试集R2：", r2_score(y_test_reverse, y_test_predict_reverse))
        print("-" * 50)
        if is_save:
            with open("./result/output.txt", "a") as f:
                f.write("-" * 20 + self.model_name + "模型评估" + "-" * 20 + "\n")
                f.write("使用特征：" + str(self.features_names) + "\n")
                f.write("训练集均方误差：" + str(mean_squared_error(y_train_reverse, y_train_predict_reverse)) + "\n")
                f.write("训练集R2：" + str(r2_score(y_train_reverse, y_train_predict_reverse)) + "\n")
                f.write("测试集均方误差：" + str(mean_squared_error(y_test_reverse, y_test_predict_reverse)) + "\n")
                f.write("测试集R2：" + str(r2_score(y_test_reverse, y_test_predict_reverse)) + "\n")
                f.write("-" * 50 + "\n")
        # 可视化
        # 标题：训练集和测试集的真实值与预测值对比
        plt.title("{}模型训练集的真实值与预测值对比".format(self.model_name))
        plt.plot(range(len(self.y_train)), y_train_predict_reverse, label="predict")
        plt.plot(range(len(self.y_train)), y_train_reverse, label="true")
        plt.legend()
        if is_save:
            plt.savefig(f"./result/{self.model_name}_train.png")
        plt.show()
        plt.title("{}模型测试集的真实值与预测值对比".format(self.model_name))
        plt.plot(range(len(self.y_test)), y_test_predict_reverse, label="predict")
        plt.plot(range(len(self.y_test)), y_test_reverse, label="true")
        plt.legend()
        if is_save:
            plt.savefig(f"./result/{self.model_name}_test.png")
        plt.show()

    def BPNN_Discharge(self, hidden_layer_sizes=(100, 100, 100), learning_rate_init=0.001, activation='relu',
                       solver='adam', is_save=False):
        """
        BP神经网络预测

        :param hidden_layer_sizes: 隐藏层神经元数目
        :param learning_rate_init: 初始学习率
        :param activation: 激活函数
        :param solver: 优化器

        :return: 测试集MSE
        """

        self.model_name = "BP神经网络"
        # 开始训练
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000,
                             learning_rate_init=learning_rate_init, activation=activation, solver=solver)
        model.fit(self.X_train, self.y_train)
        y_train_predict = model.predict(self.X_train)
        y_test_predict = model.predict(self.X_test)

        self._Visualization(y_train_predict, y_test_predict, is_save=is_save)

        if self.reverse_method is None:
            return mean_squared_error(self.y_test, y_test_predict)
        else:
            return mean_squared_error(self.reverse_method(self.y_test, self.reverse_param1, self.reverse_param2),
                                      self.reverse_method(y_test_predict, self.reverse_param1, self.reverse_param2))

    def SVM_Discharge(self, kernel='linear', C=1.0, gamma='scale', tol=1e-3, is_save=False):
        """
        SVM预测

        :param kernel: 核函数
        :param C: 惩罚系数
        :param gamma: 核系数
        :param tol: 容忍度

        :return: 测试集MSE
        """

        self.model_name = "SVM"
        # 开始训练
        model = SVR(kernel=kernel, C=C, gamma=gamma, max_iter=1000, tol=tol)
        model.fit(self.X_train, self.y_train)
        y_train_predict = model.predict(self.X_train)
        y_test_predict = model.predict(self.X_test)

        # 可视化
        self._Visualization(y_train_predict, y_test_predict, is_save=is_save)

        # 返回测试集MSE
        if self.reverse_method is None:
            return mean_squared_error(self.y_test, y_test_predict)
        else:
            return mean_squared_error(self.reverse_method(self.y_test, self.reverse_param1, self.reverse_param2),
                                      self.reverse_method(y_test_predict, self.reverse_param1, self.reverse_param2))

    def Grid_search_CV(self, model_name, cv=3, is_visual=True, max_iter=5000, verbose=0):
        """
        网格搜索交叉验证

        :param model_name: 模型名称
        :param cv: 交叉验证折数
        :param is_visual: 是否可视化

        :return: 测试集MSE
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
            model = GridSearchCV(MLPRegressor(max_iter=max_iter), param_grid, cv=cv, verbose=verbose)
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
            model = GridSearchCV(SVR(max_iter=max_iter), param_grid, cv=cv, verbose=verbose)
            model.fit(self.X_train, self.y_train)
            y_train_predict = model.predict(self.X_train)
            y_test_predict = model.predict(self.X_test)

            print(f"{self.model_name}的最优参数：{model.best_params_}")
            # 可视化
            if is_visual:
                self._Visualization(y_train_predict, y_test_predict)

        self.model = model
        if self.reverse_method is None:
            return mean_squared_error(self.y_test, y_test_predict)
        else:
            return mean_squared_error(self.reverse_method(self.y_test, self.reverse_param1, self.reverse_param2),
                                      self.reverse_method(y_test_predict, self.reverse_param1, self.reverse_param2))

def get_best_model(data, cv=3, feature_num=5, is_reverse=False, train_size=0.8, is_data_raw=True, model_save_path=None):
    """
    两种归一化方案（Z-score和min-max）、
    两种特征选择方案（Pearson相关系数和SVM法）、
    两种模型（BP神经网络和SVM），
    分别进行网格搜索交叉验证，获取最优模型

    :param data: 输入数据
    :param cv: 交叉验证折数
    :param feature_num: 特征数
    :param is_reverse: 是否进行反归一化

    :return: 最优模型的测试集MSE
    """
    best_mse = float("inf")
    best_model_name = None
    best_feature = None
    best_selector_norm = None
    best_norm = None
    if is_data_raw:
        data = add_5days_before(data)
    Normalization = [Z_score, min_max]
    reverse_Normalization = [reverse_Z_score, reverse_min_max]
    for Norm_method, reverse_method in zip(Normalization, reverse_Normalization):
        norm_data, origin_param1, origin_param2 = Norm_method(data, train_size=train_size)
        selector = [Feature_Select(), Feature_Select()]
        features = [selector[0].Pearson_Correlation2nfeature(norm_data, feature_num),
                    selector[1].SVM2nfeature(norm_data, feature_num)]
        for selector_method in selector:
            for model_name in ["SVM", "BP神经网络"]:
                print("-" * 10 + "正在进行基于{}归一化方法、{}特征选择方法、{}模型的网格搜索交叉验证".format(Norm_method.__name__,
                                                                                      selector_method.func, model_name) + "-" * 10)
                if is_reverse:
                    discharge_predict = Discharge_Predict(norm_data, selector_method.feature_result, reverse_method=reverse_method,
                                                          reverse_param=(origin_param1, origin_param2), train_size=train_size)
                else:
                    discharge_predict = Discharge_Predict(norm_data, selector_method.feature_result, train_size=train_size)
                mse_ = discharge_predict.Grid_search_CV(model_name, cv=cv, is_visual=False)
                print(f"归一化方法：{Norm_method.__name__}，特征选择方法：{selector_method.func}，"
                      f"特征数：{feature_num}，模型：{model_name}，测试集MSE：{mse_}")
                if mse_ < best_mse:
                    best_mse = mse_
                    best_model_name = model_name
                    best_feature = selector_method.feature_result
                    best_selector_norm = selector_method.func
                    best_norm = Norm_method.__name__
                    if model_save_path is not None:
                        if os.path.exists(model_save_path) is False:
                            os.makedirs(model_save_path)
                        # 保存discharge_predict.model整个模型到json文件，文件名包括归一化方法、特征选择方法、特征数、模型名
                        with open(os.path.join(model_save_path, f"{Norm_method.__name__}_{selector_method.func}_{feature_num}_{model_name}.json"), "w") as f:
                            json.dump(discharge_predict.model, f)

    print(f"最优模型：{best_model_name}，最优特征：{best_feature}，最优特征选择方法：{best_selector_norm}，"
          f"最优归一化方法：{best_norm}，测试集MSE：{best_mse}")



if __name__ == "__main__":
    ignore_warning = True
    if ignore_warning:
        import warnings
        warnings.filterwarnings("ignore")

    data = pd.read_csv("./data/01333000.csv")
    data = data.drop("Swe", axis=1)
    data = add_5days_before(data)
    data, origin_mean, origin_std = Z_score(data)
    selector = Feature_Select()
    result = selector.SVM2nfeature(data, n=5)

    discharge_predict = Discharge_Predict(data, result, reverse_method=reverse_min_max, reverse_param=(origin_mean, origin_std))
    discharge_predict.BPNN_Discharge(learning_rate_init=0.01)
    # discharge_predict.SVM_Discharge(C=1, gamma='scale', tol=0.001)

    # import time
    # print("开始时间: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # data = pd.read_csv("./data/01333000.csv")
    # data = data.drop("Swe", axis=1)
    # get_best_model(data, cv=3, feature_num=5, is_reverse=False)
    # print("结束时间: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

