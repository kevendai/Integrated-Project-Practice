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
# Date: 日期
# Discharge: 流量
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
from utils.data_process import read_from_dataset_folders, add_5days_before

def data_analysis(data):
    # 可视化数据分布
    pd.set_option('display.max_columns', None) # 显示所有列
    print("")
    print("-" * 20 + "数据描述：" + "-" * 20)
    print(data.describe()) # 包括计数、均值、标准差、最小值、四分位数、最大值
    print("-" * 20 + "数据信息：" + "-" * 20)
    print(data.info()) # 包括每列的非空值数量、数据类型
    print("-" * 20 + "数据列名：" + "-" * 20)
    print(data.columns) # 列名

    # 直方图
    # 显示图表名称
    data.hist(bins=50, figsize=(20, 15))
    plt.title("Histogram")
    plt.show()

    # 异常值检测
    # 是否有缺失值
    print("-" * 20 + "缺失值检测：" + "-" * 20)
    print("是否有缺失值：", data.isnull().values.any())
    if data.isnull().values.any():
        print("缺失值数量：", data.isnull().sum().sum())
        print("每列缺失值数量：")
        print(data.isnull().sum())
    # 是否有全为零的列或者行
    print("-" * 20 + "全为零检测：" + "-" * 20)
    print("是否有全为零的列：", (data == 0).all().any())
    if (data == 0).all().any():
        print("全为零的列数量：", (data == 0).all().sum())
        print("全为零的列名：")
        print(data.columns[(data == 0).all()])
    print("是否有全为零的行：", (data == 0).all(axis=1).any())
    if (data == 0).all(axis=1).any():
        print("全为零的行数量：", (data == 0).all(axis=1).sum())
        print("全为零的行索引：")
        print(data.index[(data == 0).all(axis=1)])

    # 相关性矩阵
    corr_matrix = data.iloc[:, 1:].corr()
    print("-" * 20 + "相关性矩阵：" + "-" * 20)
    print(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu") # RdYlGn
    plt.title("Correlation Matrix")
    plt.show()

    # Discharge与其他变量的散点图
    sns.pairplot(data, y_vars=["Discharge"], x_vars=['Prcp'], kind='scatter')
    plt.title("Scatter Plot")
    plt.show()

    # Discharge随时间的折线图
    data.plot(x='Date', y='Discharge', kind='line')
    plt.title("Discharge Line Plot")
    plt.show()

    # 所有数据的箱线图
    data.boxplot()
    plt.title("Boxplot")
    plt.show()


if __name__ == "__main__":
    data = read_from_dataset_folders(drop_Swe=False)
    data = add_5days_before(data)
    data_analysis(data)
