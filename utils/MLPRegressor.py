#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：MLPRegressor.py
@Author  ：Dai Yikai
@Date    ：2024/4/9 22:36 
@Function：神经网络GPU加速，本来想加速的，发现还是很慢，就放这里了，不用管
"""
import torch
import torch.nn as nn
class MLPRegressor(nn.Module):
    """
    多层感知机回归模型

    Attributes:
        hidden_layer_sizes: 隐藏层大小，默认为100
        max_iter: 最大迭代次数，默认为200
        learning_rate_init: 初始学习率，默认为0.001
        activation: 激活函数，默认为"relu"
        solver: 优化器，默认为"adam"
        layers: 神经网络层

    """
    def __init__(self,
                 hidden_layer_sizes=(100, ),
                 max_iter=200,
                 learning_rate_init=0.001,
                 activation="relu",
                 solver="adam",
                 batch_size=32,
    ):
        super(MLPRegressor, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.activation = activation
        self.solver = solver
        self.layers = nn.Sequential()
        self.batch_size = batch_size

        self.layers.append(nn.Linear(1, self.hidden_layer_sizes[0]))
        if self.activation == "relu":
            self.layers.append(nn.ReLU())
        elif self.activation == "tanh":
            self.layers.append(nn.Tanh())
        elif self.activation == "logistic":
            self.layers.append(nn.Sigmoid())
        for i in range(1, len(self.hidden_layer_sizes)):
            self.layers.append(nn.Linear(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
            if self.activation == "relu":
                self.layers.append(nn.ReLU())
            elif self.activation == "tanh":
                self.layers.append(nn.Tanh())
            elif self.activation == "logistic":
                self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(self.hidden_layer_sizes[-1], 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

    def init_layers(self, input_size):
        # 替换第一层的输入大小
        self.layers[0] = nn.Linear(input_size, self.hidden_layer_sizes[0])

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        y = y.reshape(-1, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # init_layers
        self.init_layers(X.shape[1])
        if self.solver == "adam":
            optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate_init)
        elif self.solver == "sgd":
            optimizer = torch.optim.SGD(self.layers.parameters(), lr=self.learning_rate_init)
        else:
            raise Exception("不支持的优化器")
        loss_func = nn.MSELoss()
        self.layers.to(self.device)
        self.to(self.device)
        last_loss = float("inf")
        epoch_loss = 0
        for epoch in range(self.max_iter):
            for batch_X, batch_y in dataloader:
                output = self.forward(batch_X.to(self.device))
                loss = loss_func(output, batch_y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    epoch_loss += loss.item()
            if abs(last_loss - epoch_loss) < 1e-6:
                break
            with torch.no_grad():
                last_loss = epoch_loss
                epoch_loss = 0
            print("epoch: {}, loss: {}".format(epoch, last_loss))


    def predict(self, X):
        data = torch.tensor(X, dtype=torch.float32)
        data = data.to(self.device)
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            y_pred = []
            for batch_X in dataloader:
                y_pred.append(self.forward(batch_X[0]))
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.cpu().numpy()
        return y_pred

    def score(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        test_dataset = torch.utils.data.TensorDataset(X, y)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
        # 计算R2
        with torch.no_grad():
            y_pred = []
            for batch_X, batch_y in test_dataloader:
                y_pred.append(self.forward(batch_X.to(self.device)).cpu())
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.numpy()
            y = y.reshape(-1, 1)
            y = y.numpy()

            r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return r2


class Grid_search_CV_for_MLP():
    """
    网格搜索交叉验证，专门用于MLPRegressor

    Attributes:
        param_grid: 参数网格，为字典，Key为参数名，Value为参数列表
        test_size: 测试集划分系数
        scoring: 评分函数
        best_params_: 最佳参数
        best_score_: 最佳分数
    """
    def __init__(self, param_grid, test_size=0.2):
        self.param_grid = param_grid
        self.test_size = test_size
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X, y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)
        param_dict_list = self.generate_param_dict(self.param_grid)
        for param_dict in param_dict_list:
            mlp = MLPRegressor(**param_dict)
            mlp.fit(X_train, y_train)
            mlp_score = mlp.score(X_test, y_test)
            print("参数：", param_dict)
            print("分数：", mlp_score)
            if self.best_score_ is None or mlp_score > self.best_score_:
                self.best_score_ = mlp_score
                self.best_params_ = param_dict

    def generate_param_dict(self, param_grid):
        """
        生成一个param_dict_list，包含所有参数组合的字典，用于传入MLPRegressor

        :param param_grid: 是一个字典，Key为参数名，Value为参数列表
        :return: 参数字典列表
        """
        param_dict_list = []
        for key in param_grid.keys():
            if not param_dict_list:
                for value in param_grid[key]:
                    param_dict_list.append({key: value})
            else:
                temp_param_dict_list = []
                for value in param_grid[key]:
                    for param_dict in param_dict_list:
                        new_param_dict = param_dict.copy()
                        new_param_dict[key] = value
                        temp_param_dict_list.append(new_param_dict)
                param_dict_list = temp_param_dict_list
        return param_dict_list

if __name__ == "__main__":
    import numpy as np

    # param_grid = {
    #     'hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), (150, 150, 150)],
    #     'learning_rate_init': [0.001, 0.01, 0.1],
    #     'activation': ['relu', 'logistic', 'tanh'],
    #     'solver': ['adam', 'sgd']
    # }
    # net = Grid_search_CV_for_MLP(param_grid)
    # list_param = net.generate_param_dict(param_grid)
    # for i in range(len(list_param)):
    #     print(list_param[i])
    # print(len(list_param))


    import pandas as pd

    data = pd.read_csv("../data/01333000.csv")
    data = data.drop("Swe", axis=1)
    from data_process import add_5days_before, min_max
    from feature_select import Feature_Select
    data = add_5days_before(data)
    data, origin_mean, origin_std = min_max(data)
    selector = Feature_Select()
    result = selector.SVM2nfeature(data, n=5)
    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), (150, 150, 150)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'activation': ['relu', 'logistic', 'tanh'],
        'solver': ['adam', 'sgd']
    }
    X = data.loc[:, result]
    y = data["Discharge"]
    # dataframe转numpy
    X = X.values
    y = y.values

    discharge_predict = Grid_search_CV_for_MLP(param_grid)
    discharge_predict.fit(X, y)
    print("最佳参数：", discharge_predict.best_params_)
    print("最佳分数：", discharge_predict.best_score_)