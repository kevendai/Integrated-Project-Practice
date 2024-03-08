```
Python 3.9.18
pandas 2.2.1
numpy 1.26.4
pillow 10.2.0
seaborn 0.13.2
scipy 1.12.0
scikit-learn 1.4.1.post1
```

### 2024-3-7 21:41 已完成数据可视化分析，utils.data_process中有添加五天内的径流量数据的函数
### 2024-3-7 21:57 已完成Z-score标准化与min-max标准化，在utils.data_process中；在初始读入时删去swe（似乎在所有数据中均为零）
### 2024-3-8 00:06 已完成特征筛选，皮尔逊和最大系数法，在utils.feature_select中
### 2024-3-8 16:47 封装特征选择类，皮尔逊相关系数法、最大信息系数法、距离相关系数法