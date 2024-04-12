#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：综合项目实践 
@File    ：Basin_Clustering.py
@Author  ：Dai Yikai
@Date    ：2024/4/12 16:42 
@Function：
"""
import pandas as pd
import os
import numpy as np
from data_process import Z_score, min_max

class Basin_Clustering:
    """
    流域聚类类

    Attributes:
        data_root_path: 数据路径，默认为"./data/dataset02"
        train_size: 训练集大小，默认为1.0
        Clustering_result: 聚类结果
        feature_dict: 特征字典
        Clustering_func: 聚类方法
        train_size: 训练集大小，默认为1.0
        Normalize_func: 归一化方法，默认为"min_max"，可选"Z_score"

    """
    def __init__(self, root_path="./data/dataset02", train_size=1.0, Normalize_func="min_max", is_PCA=True):
        self.data_root_path = root_path
        self.train_size = 1.0
        self.Clustering_result = {}
        self.feature_dict = None
        self.Clustering_func = None
        self.train_size = train_size
        self.Normalize_func = Normalize_func
        self.K = None
        self.SSE_list = None
        self.is_PCA = is_PCA

    def csv2feature(self, data_root):
        """
        读取csv文件，提取特征

        Args:
            data_root: 数据路径

        Returns:
            data: 特征

        """
        data = pd.read_csv(data_root).drop(columns=["Swe"])

        data["Date"] = data["Date"].apply(lambda x: x[:7])
        data = data.groupby("Date").mean()
        data = data.reset_index()

        mean_df = data.iloc[:, 1:].mean()
        std_df = data.iloc[:, 1:].std()

        # 归一化
        if self.Normalize_func == "Z_score":
            data, _1, _2 = Z_score(data, train_size=self.train_size)
        elif self.Normalize_func == "min_max":
            data, _1, _2 = min_max(data, train_size=self.train_size)
        else:
            raise Exception("不支持的归一化方法")

        # 增加频域特征
        data["Discharge_FFT"] = np.real(np.fft.fft(data["Discharge"].values))
        # 删除Date
        data.drop(columns=["Date"], inplace=True)
        data = np.append(data, mean_df.values)
        data = np.append(data, std_df.values)
        data = data.reshape(-1)

        return data

    def csvfiles2features(self):
        """
        读取文件夹下所有csv文件，提取特征
        Args:

        Returns:
            feature_dict: 特征字典

        """
        feature_dict = {}
        for file in os.listdir(self.data_root_path):
            data = self.csv2feature(os.path.join(self.data_root_path, file))
            feature_dict[file] = data

        self.feature_dict = feature_dict

    def PCA4Clustering(self, n_components=5):
        """
        特征太多，先进行PCA降维
        Args:
            n_components: 降维后的维度，默认为5

        Returns:

        """
        from sklearn.decomposition import PCA
        data = np.array(list(self.feature_dict.values()))
        pca = PCA(n_components=n_components)
        pca.fit(data)
        data = pca.transform(data)
        # 降维结果重新写到feature_dict
        for i in range(len(self.feature_dict.keys())):
            self.feature_dict[list(self.feature_dict.keys())[i]] = data[i]

    def KMeans(self, n_clusters=3):
        """
        KMeans聚类
        Args:
            n_clusters: 聚类数，默认为3

        Returns:

        """
        from sklearn.cluster import KMeans
        data = np.array(list(self.feature_dict.values()))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)

        return kmeans.labels_

    def DBSCAN(self, eps=0.5, min_samples=5):
        """
        DBSCAN聚类
        Args:
            eps: 半径，默认为0.5
            min_samples: 最小样本数，默认为5

        Returns:

        """
        from sklearn.cluster import DBSCAN
        data = np.array(list(self.feature_dict.values()))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)

        return dbscan.labels_

    def AGNES(self, n_clusters=3):
        """
        AGNES聚类
        Args:
            n_clusters: 聚类数，默认为3

        Returns:

        """
        from sklearn.cluster import AgglomerativeClustering
        data = np.array(list(self.feature_dict.values()))
        agnes = AgglomerativeClustering(n_clusters=n_clusters)
        agnes.fit(data)

        return agnes.labels_

    def fit(self, n_clusters=3, n_components=5, Clustering_func="KMeans", eps=None, min_samples=None):
        """
        聚类

        Args:
            n_clusters: 聚类数，默认为3
            n_components: 降维后的维度，默认为5
            Clustering_func: 聚类方法，默认为"KMeans"，可选"DBSCAN"、"AGNES"
            eps: DBSCAN半径，默认为None
            min_samples: DBSCAN最小样本数，默认为None

        Returns:

        """
        self.K = n_clusters
        if self.feature_dict is None:
            self.csvfiles2features()
            if self.is_PCA:
                self.PCA4Clustering(n_components=n_components)
        if Clustering_func == "KMeans":
            labels = self.KMeans(n_clusters=n_clusters)
        elif Clustering_func == "DBSCAN" and eps is not None and min_samples is not None:
            labels = self.DBSCAN(eps=eps, min_samples=min_samples)
        elif Clustering_func == "AGNES":
            labels = self.AGNES(n_clusters=n_clusters)
        else:
            raise Exception("不支持的聚类方法，或缺少参数")
        for i in range(len(self.feature_dict.keys())):
            self.Clustering_result[list(self.feature_dict.keys())[i]] = labels[i]

    # 肘部法
    def elbow_method(self, n_components=5, max_k=20):
        """
        肘部法确定最佳聚类数

        Args:
            n_components: 降维后的维度，默认为5
            max_k: 最大聚类数，默认为20

        Returns:

        """
        self.SSE_list = []
        for i in range(2, max_k):
            self.fit(n_clusters=i, n_components=n_components)
            self.SSE_list.append(self.SSE())
        # 画图
        import matplotlib.pyplot as plt
        plt.plot(range(2, max_k), self.SSE_list, "o-")
        plt.xlabel("K")
        plt.ylabel("SSE")
        plt.show()

    def SSE(self):
        """
        计算SSE

        Args:

        Returns:
            SSE: SSE

        """
        data = np.array(list(self.feature_dict.values()))
        labels = np.array(list(self.Clustering_result.values()))
        centers = []
        for i in range(len(np.unique(labels))):
            centers.append(data[labels == i].mean(axis=0))
        centers = np.array(centers)
        SSE = 0
        for i in range(len(data)):
            SSE += np.linalg.norm(data[i] - centers[labels[i]])
        return SSE



if __name__ == "__main__":
    bc = Basin_Clustering("../data/dataset02", Normalize_func="Z_score")
    # bc.elbow_method(max_k=15, n_components=5)
    bc.fit(n_clusters=8, n_components=5)

    with open("../data/camels_name.txt", "r") as f:
        camels_name = f.readlines()
    camels_name = camels_name[1:]
    set_name = {}
    for i in range(len(camels_name)):
        set_name[camels_name[i][:8]] = (camels_name[i].strip("\n")[-2:])
    print(set_name)

    for i in range(len(set_name.keys())):
        set_name[list(set_name.keys())[i]] = [list(set_name.values())[i], bc.Clustering_result[list(set_name.keys())[i]+".csv"]]

    set_name = dict(sorted(set_name.items(), key=lambda x: x[1][1]))
    for i in range(len(set_name.keys())):
        print("流域{}的聚类结果为：{}".format(list(set_name.keys())[i], list(set_name.values())[i]))