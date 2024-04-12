#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：data_process.py
@Author  ：Dai Yikai
@Date    ：2024/3/7 20:36 
@Function：读取文件、添加前五天数据
"""

import pandas as pd
import os
import copy
import numpy as np


def read_from_dataset_folders(path="./data/dataset02", cal_avg=True, drop_Swe=True):
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

    if drop_Swe:
        data = data.drop("Swe", axis=1)

    return data


def read_from_list(data_list, drop_Swe=True, is_add_Prcp=True):
    """
    通过列表读取文件

    :param data_list:输入列表数据
    :param drop_Swe:是否删除Swe列
    :return:数据
    """
    data = pd.read_csv(data_list[0])
    if drop_Swe:
        data = data.drop("Swe", axis=1)
    data = add_5days_before(data, is_add_Prcp=is_add_Prcp)
    for i in range(1, len(data_list)):
        if data_list[i].split(".")[-1] != "csv":
            data_list[i] = data_list[i] + ".csv"
        new_data = pd.read_csv(data_list[i])
        if drop_Swe:
            new_data = new_data.drop("Swe", axis=1)
        new_data = add_5days_before(new_data, is_add_Prcp=is_add_Prcp)
        data = pd.concat([data, new_data], axis=0)
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data


def read_from_csv(path="./data/01333000.csv", drop_Swe=True, is_add_Prcp=True):
    """
    通过csv文件读取数据

    :param path:输入文件路径
    :param drop_Swe:是否删除Swe列
    :return:数据
    """
    data = pd.read_csv(path)
    if drop_Swe:
        data = data.drop("Swe", axis=1)
    data = add_5days_before(data, is_add_Prcp=is_add_Prcp)
    return data


def add_5days_before(data, is_add_Prcp=True):
    """
    加入前五天的数据，同时删除最早五天的数据

    :param data:输入不少于5天的数据
    :param is_add_Prcp:是否加入降水量
    :return:加入5天前的数据，数据标签为one_Discharge, two_Discharge, three_Discharge, four_Discharge, five_Discharge，
    如果加入降水量，标签为one_Prcp, two_Prcp, three_Prcp, four_Prcp, five_Prcp
    """
    if len(data) < 5:
        raise Exception("数据长度不足5天")

    add_col_name = ["one_Discharge", "two_Discharge", "three_Discharge", "four_Discharge", "five_Discharge"]

    if is_add_Prcp:
        # 添加五天的降水量
        add_col_name += ["one_Prcp", "two_Prcp", "three_Prcp", "four_Prcp", "five_Prcp"]

    for i in range(len(add_col_name)):
        feature_name = add_col_name[i].split("_")[1]
        data[add_col_name[i]] = data[feature_name].shift(i % 5 + 1)

    data = data.dropna()
    return data


def Z_score(data, train_size=0.8):
    """
    使用Z_score方法标准化数据

    :param data:输入数据
    :return:标准化后的数据
    """
    new_df = pd.DataFrame(columns=data.columns)
    new_df["Date"] = data["Date"]
    mean = data.iloc[:int(len(data) * train_size), 1:].mean()
    std = data.iloc[:int(len(data) * train_size), 1:].std()
    for col in data.columns[1:]:
        new_df[col] = (data[col] - mean[col]) / (std[col] + 1e-8)
    return new_df, data["Discharge"].mean(), data["Discharge"].std()


def min_max(data, train_size=0.8):
    """
    使用min-max方法标准化数据

    :param data:输入数据
    :return:标准化后的数据
    """
    new_df = pd.DataFrame(columns=data.columns)
    new_df["Date"] = data["Date"]
    max_value = data.iloc[:int(len(data) * train_size), 1:].max()
    min_value = data.iloc[:int(len(data) * train_size), 1:].min()
    for col in data.columns[1:]:
        new_df[col] = (data[col] - min_value[col]) / (max_value[col] - min_value[col] + 1e-8)
    return new_df, data["Discharge"].min(), data["Discharge"].max()


# Z-score简单归一化
def Z_score_simple(data):
    """
    使用Z_score方法标准化数据

    :param data:输入数据
    :return:标准化后的数据
    """
    new_df = pd.DataFrame(columns=data.columns)
    new_df["Date"] = data["Date"]
    for col in data.columns[1:]:
        new_df[col] = 1 / (data[col] + 1)
    return new_df


# 对数归一化
def log_normalization(data):
    """
    使用log方法标准化数据

    :param data:输入数据
    :return:标准化后的数据
    """
    new_df = pd.DataFrame(columns=data.columns)
    new_df["Date"] = data["Date"]
    for col in data.columns[1:]:
        new_df[col] = - np.log(data[col] + 1)
    return new_df


def reverse_min_max(data, origin_min, origin_max):
    """
    使用min-max方法反标准化数据，数据为单独的一列

    :param data:输入数据
    :param origin_min:最小值
    :param origin_max:最大值
    """

    new_df = copy.deepcopy(data) * (origin_max - origin_min) + origin_min
    return new_df


def reverse_Z_score(data, origin_mean, origin_std, ):
    """
    使用Z_score方法反标准化数据，数据为单独的一列

    :param data:输入数据
    :param origin_mean:均值
    :param origin_std:标准差
    """
    new_df = copy.deepcopy(data) * origin_std + origin_mean
    return new_df


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    test_data = read_from_dataset_folders("../data/dataset02")
    print(test_data)
    data = add_5days_before(test_data)
    print(data)
    data = min_max(data)
    print(data)
