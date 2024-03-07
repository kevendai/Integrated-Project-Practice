#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：data_process.py
@Author  ：Dai Yikai
@Date    ：2024/3/7 20:36 
@Function：
"""

import pandas as pd
import os

def read_from_dataset_folders(path="./data/dataset02", cal_avg=True):
    # 读取文件夹下所有文件
    file_list = os.listdir(path)


    # 异常处理：非csv文件、空文件夹
    for file in file_list:
        if file.split(".")[-1] != "csv":
            file_list.remove(file)
    file_num = len(file_list)
    if file_num == 0:
        raise Exception("文件夹为空")

    # 读取所有文件，并相加，但是data字串不相加
    data = pd.read_csv(path + "/" + file_list[0])
    for i in range(1, file_num):
        data.iloc[:, 1:] += pd.read_csv(path + "/" + file_list[i]).iloc[:, 1:]

    # 计算平均值
    if cal_avg:
        data[data.columns[1:]] = data[data.columns[1:]] / file_num

    return data

def add_5days_before(data):
    """
    加入前五天的数据，同时删除最早五天的数据

    :param data:输入不少于5天的数据
    :return:加入5天前的数据，数据标签为onedb, twodb, threedb, fourdb, fivedb
    """

    add_col_name = ["onedb", "twodb", "threedb", "fourdb", "fivedb"]

    for i in range(len(add_col_name)):
        data.loc[:, add_col_name[i]] = data["Discharge"].shift(i+1)

    data = data.dropna()
    return data

if __name__ == "__main__":
    test_data = read_from_dataset_folders("../data/dataset02")
    print(test_data)
    data = add_5days_before(test_data)
    print(data)