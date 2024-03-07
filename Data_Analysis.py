#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践
@File    ：Data_Analysis.py
@Author  ：Dai Yikai
@Date    ：2024/3/7 17:34
@Function：对数据进行统计分析和探索性数据分析，并进行可视化，深入了解数据分布、异常及相关性等情况
"""

# 数据包括Date,Discharge,Dayl,Prcp,Srad,Swe,Tmax,Tmin,Vp字段
# Date: 日期, Discharge: 流量
# Dayl: 每天白天的持续时间（以秒为单位）。此计算基于一天中太阳位于假设的平坦地平线上方的时间段
# Prcp: 每日总降水量（毫米）。所有形式降水的总和转换为水当量深度。
# Srad: 入射短波辐射通量密度（以瓦/平方米为单位），取一天中白天时段的平均值。注：每日总辐射量（MJ/m2/day）可计算如下：（（srad (W/m2) * dayl (s/day)）/l,000,000）
# Swe:  雪水当量，单位为千克每平方米。积雪中所含的水量。
# Tmax: 每日最高 2 m 气温（摄氏度）。
# Tmin: 每日最低 2 m 气温（摄氏度）。
# Vp:   水蒸气压（以帕斯卡为单位）。日平均水蒸气分压。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def data_analysis(data):
    # 可视化数据分布
    print(data.describe())
    print(data.info())


data = read_from_dataset_folders()
data_analysis(data)
