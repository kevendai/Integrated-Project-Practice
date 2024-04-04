#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：feature_select.py
@Author  ：Dai Yikai
@Date    ：2024/3/7 22:54 
@Function：在这里我们举出了四种特征选择方法，分别是皮尔逊相关系数法、最大信息系数法、距离相关系数法和SVM法
"""

import pandas as pd
import numpy as np

class Feature_Select:
    """特征选择类，选择与Discharge相关性最高的n个特征

    皮尔逊相关系数法: Pearson_Correlation2nfeature；
    最大信息系数法: MIC2nfeature；
    距离相关系数法: Distance_Correlation2nfeature；
    SVM法: SVM2nfeature。

    Attributes:
        n: 选择的特征数，初始化为None，调用方法后赋值
        func: 选择特征的方法，初始化为None，调用方法后赋值
        feature_result: 选择的特征，初始化为None，调用方法后赋值

    """
    def __init__(self):
        self.n = None
        self.func = None
        self.feature_result = None

    def Pearson_Correlation2nfeature(self, data, n=5):
        """
        选择与Discharge相关性最高的n个特征

        :param data: 输入数据
        :param n: 选择的特征数
        :return: 选择的特征
        """
        self.func = "皮尔逊相关系数法"
        self.n = n
        corr_matrix = data.iloc[:, 1:].corr()
        corr_matrix = abs(corr_matrix["Discharge"]).sort_values(ascending=False)
        self.feature_result = corr_matrix.index[1:n+1]
        return self.feature_result

    def MIC2nfeature(self, data, n=5):
        """
        最大信息系数法选择与Discharge相关性最高的n个特征

        :param data: 输入数据
        :param n: 选择的特征数
        :return: 选择的特征
        """
        # 在https://www.lfd.uci.edu/~gohlke/pythonlibs/#minepy 下载minepy-1.2.6-cp39-cp39-win_amd64.whl
        from minepy import MINE
        self.func = "最大信息系数法"
        self.n = n
        mine = MINE()
        mic = []
        # 跳过data的第一列
        for col in data.columns[1:]:
            mine.compute_score(data[col], data["Discharge"])
            mic.append(mine.mic())
        mic = np.array(mic)
        self.feature_result = data.columns[1:][np.argsort(-mic)][1:n+1]
        return self.feature_result

    def Distance_Correlation2nfeature(self, data, n=5, distance_type="correlation"):
        """
        距离相关系数法选择与Discharge相关性最高的n个特征

        :param data: 输入数据
        :param n: 选择的特征数
        :param distance_type: 距离类型：chebyshev（切比雪夫距离）、cityblock（曼哈顿距离）、correlation（相关系数）、cosine（余弦距离）、euclidean（欧氏距离）、sqeuclidean（欧氏距离的平方）
        :return: 选择的特征
        """
        from scipy.spatial import distance
        self.func = "距离相关系数法"
        self.n = n
        dis_corr = []
        for col in data.columns[1:]:
            dis_corr.append(eval("distance." + distance_type)(data[col], data["Discharge"]))
        dis_corr = np.array(dis_corr)
        self.feature_result = data.columns[1:][np.argsort(-dis_corr)][1:n+1]
        return self.feature_result

    def SVM2nfeature(self, data, n=5):
        """
        SVM法选择与Discharge相关性最高的n个特征

        :param data: 输入数据
        :param n: 选择的特征数
        :return: 选择的特征
        """
        from sklearn.svm import SVR
        self.func = "SVM法"
        self.n = n
        X = data.iloc[:, 2:]
        y = data["Discharge"]
        clf = SVR(kernel='linear')
        clf.fit(X, y)
        # clf.coef_是一个ndarray，存放了每个特征的权重
        self.feature_result = X.columns[np.argsort(-abs(clf.coef_))[0]][:n]

        return self.feature_result


if __name__ == "__main__":
    from data_process import read_from_dataset_folders, add_5days_before, Z_score
    data = read_from_dataset_folders("../data/dataset02")
    data = add_5days_before(data)
    data = Z_score(data)
    selector = Feature_Select()
    result = selector.SVM2nfeature(data)
    print("使用{}方法筛选出的特征为：{}".format(selector.func, result))