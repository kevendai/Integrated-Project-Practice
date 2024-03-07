#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：feature_select.py
@Author  ：Dai Yikai
@Date    ：2024/3/7 22:54 
@Function：
"""

import pandas as pd
import numpy as np

def Pearson_Correlation2nfeature(data, n=5):
    """
    选择与Discharge相关性最高的n个特征

    :param data: 输入数据
    :param n: 选择的特征数
    :return: 选择的特征
    """
    corr_matrix = data.corr()
    corr_matrix = abs(corr_matrix["Discharge"]).sort_values(ascending=False)
    return corr_matrix.index[1:n+1]

def MIC2nfeature(data, n=5):
    """
    最大信息系数法选择与Discharge相关性最高的n个特征

    :param data: 输入数据
    :param n: 选择的特征数
    :return: 选择的特征
    """
    # 在https://www.lfd.uci.edu/~gohlke/pythonlibs/#minepy 下载minepy-1.2.6-cp39-cp39-win_amd64.whl
    from minepy import MINE
    mine = MINE()
    mic = []
    # 跳过data的第一列
    for col in data.columns[1:]:
        mine.compute_score(data[col], data["Discharge"])
        mic.append(mine.mic())
    mic = np.array(mic)
    return data.columns[1:][np.argsort(-mic)][1:n+1]

if __name__ == "__main__":
    from data_process import read_from_dataset_folders, add_5days_before, Z_score
    data = read_from_dataset_folders("../data/dataset02")
    data = add_5days_before(data)
    data = Z_score(data)
    print(MIC2nfeature(data))