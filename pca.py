'''
Descripttion: 
Version: 1.0
Author: xiawei
Date: 2022-09-26 14:33:15
LastEditors: xiawei
LastEditTime: 2022-12-18 10:49:57
'''
'''
1.基于iris实现knn分类，neighbors=3
2.对数据标准化，选一个维度可视化
3.对原数据等维度pca，查看各主成分方差比例
4.保留合适的主成分(principle components)，可视化降维之后的数据
5.基于降维之后的数据knn，与原数据对比
'''

from sklearn.metrics import accuracy_score
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
current_directory = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(current_directory+'.\iris_data.csv')
# print(data.head())

X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
# print("X", X, "y", y)

KNN = KNeighborsClassifier(n_neighbors=3)

print("X", X.shape, "y", y.shape)
KNN.fit(X, y)
y_predict = KNN.predict(X)
knn_score = accuracy_score(y, y_predict)
print('knn_score', knn_score)


# 标准化数据
X_norm_data = StandardScaler().fit_transform(X)
# print('X_norm_data', X_norm_data)
# 找一个维度的数据看原数据和标准化之后的数据的概率分布，能得到标准化的直观结果
fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(X.loc[:, 'sepal length'], bins=100)
plt.subplot(122)
plt.hist(X_norm_data[:, 0], bins=100)
# plt.show()
# 直观看到均值方差不同，继续用mean和std看一下均值和方差具体值
x1_mean = data.loc[:, 'sepal length'].mean()
x1_sigma = data.loc[:, 'sepal length'].std()
x1_norm_mean = X_norm_data[:, 0].mean()
x1_norm_sigma = X_norm_data[:, 0].std()
print('x1_mean', x1_mean, 'x1_sigma', x1_sigma,
      'x1_norm_mean', x1_norm_mean, 'x1_norm_sigma', x1_norm_sigma)


# 对原数据等维度pca
print(X.shape[1])
pca = PCA(X.shape[1])
X_pca = pca.fit_transform(X_norm_data)
# 计算X_pca即降维之后的数据，观察方差，方差大的数据有效，方差小的说明数据本身相关性大，不是主成分。
var_ratio = pca.explained_variance_ratio_
# [0.72770452 0.23030523 0.03683832 0.00515193]
# print(var_ratio)
fig3 = plt.figure(figsize=(20, 5))
plt.bar([1, 2, 3, 4], var_ratio)
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4'])
plt.show()

# 只取原数据的2维度进行pca
pca = PCA(2)
X_pca = pca.fit_transform(X_norm_data)
print(X_pca.shape)
# 对二维主成分进行画图
fig4 = plt.figure(figsize=(10, 5))
# 加上标签 y=0，1，2
xa = plt.scatter(X_pca[:, 0][y == 0], X_pca[:, 1][y == 0])
xb = plt.scatter(X_pca[:, 0][y == 1], X_pca[:, 1][y == 1])
xc = plt.scatter(X_pca[:, 0][y == 2], X_pca[:, 1][y == 2])
plt.legend([xa, xb, xc], ['xa', 'xb', 'xc'])
plt.show()

# 对降维之后的数据进行knn建模
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_pca, y)
y_predict = KNN.predict(X_pca)
knn_pca_score = accuracy_score(y, y_predict)
print('y_predict', y_predict)
# print('knn_pca_score', knn_pca_score)
