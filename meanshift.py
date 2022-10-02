"""查找轮廓，在原图上绘制轮廓 demo1"""

'''import cv2
import numpy as np
img = cv2.imread('test.png')
sp = 20
sr = 30
mean_img = cv2.pyrMeanShiftFiltering(img, sp, sr)

canny_img = cv2.Canny(mean_img, 150, 300)

# 查找轮廓api，原始图像，查找轮廓方式RETR_EXTERNAL为之查找外围，List从里到外从右到左，CCOMP有层级关系没搞懂
# 从大到小从右到左,这里的contours为根据CHAIN_APPROX_SIMPLE设定的点。
contours, _ = cv2.findContours(
    canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓，-1为绘制所有轮廓，颜色，线宽。
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.imshow('mean_img', mean_img)
cv2.imshow('canny_img', canny_img)
cv2.waitKey()'''


''' 采用meanshift对2d数据聚类，并预测（80，60）的类别，计算准确率'''

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift, estimate_bandwidth
data = pd.read_csv('data.csv')
print(data.head())
X = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']
# establish a KNN model
# KNN为监督式模型，必须告知输入是几类
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X, y)
y_predict_KNN_test = KNN.predict([[80, 60]])
print('y_predict_KNN_test', y_predict_KNN_test)
y_predict_knn = KNN.predict(X)
knn_score = accuracy_score(y, y_predict_knn)
print('knn_score', knn_score)

# 下面是mean shift
# 用estimate_bandwidth
bw = estimate_bandwidth(X, n_samples=500)
print(bw)
MS = MeanShift(bw)
MS.fit(X)
y_predict_ms = MS.predict(X)
print(pd.value_counts(y_predict_ms), pd.value_counts(y))
# 此时的预测结果和原始数据类别标签对不上，需要手动更正标签
