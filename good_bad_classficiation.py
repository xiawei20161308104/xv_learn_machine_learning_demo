"""
基于data_class_raw.csv数据，根据高斯密度概率函数，寻找异常点并剔除。
基于data_class_processed.csv数据，进行pca处理，确定重要数据维度及成分
完成数据分离，参数为random_state=4,test_size=0.4
建立knn模型完成分类，邻居选择10，计算准确率可视化边界
计算测试数据集对应的混淆矩阵，计算准确率召回率特异度精确率f1分数
尝试改变邻居参数，计算其在训练集测试集上的准确率并画出
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# covariance标准差 异常检测包
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 混淆矩阵包
from sklearn.metrics import confusion_matrix


current_directory = os.path.dirname(os.path.abspath(__file__))
# print(current_directory)
# load the data,define the data, visualize the data
data = pd.read_csv(current_directory+'/'+'data_class_raw.csv')
# print(data.head())
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']
# print('X', X, 'y', y)
fig0 = plt.figure(figsize=(5, 5))
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.legend((bad, good), ('bad', 'good'))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
# plt.show()

# anomay detection
AD = EllipticEnvelope(contamination=0.02)
AD.fit(X[y == 0])
bad_AD_predict = AD.predict(X[y == 0])
# print('bad_AD_predict', bad_AD_predict)
# visualize the anomay data
fig1 = plt.figure(figsize=(10, 10))
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
anomay_bad = plt.scatter(X.loc[:, 'x1'][y == 0]
                         [bad_AD_predict == -1], X.loc[:, 'x2'][y == 0][bad_AD_predict == -1], marker='x', s=150)
plt.legend((bad, good, anomay_bad), ('bad', 'good', 'anomay_bad'))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('data with anomay_bad')
# plt.show()
# 以上结束第一个任务，开始第二个


data = pd.read_csv(current_directory+'/'+'data_class_processed.csv')
# print(data.head())
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']
# print('X', X, 'y', y)

# pca
X_norm = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_reduction = pca.fit_transform(X_norm)
# show the 标准差比率，观测主成分选择个数
covariance_ratio = pca.explained_variance_ratio_
# covariance_ratio [0.5369408 0.4630592]  两个都是主成分
# print('covariance_ratio', covariance_ratio)
# visualize the ratio
fig2 = plt.figure(figsize=(5, 5))
plt.bar([1, 2], covariance_ratio)
# plt.show()


# train and test split:random_state=4,test_size=0.4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=4, test_size=0.4)
# (21, 2) (14, 2) (35, 2) 将数据6：4分割，纵维度不变。
# print(X_train.shape, X_test.shape, X.shape)


# 建立knn模型完成分类，邻居选择10，计算准确率可视化边界
KNN_10 = KNeighborsClassifier(n_neighbors=10)
KNN_10.fit(X_train, y_train)
y_train_predict = KNN_10.predict(X_train)
y_test_predict = KNN_10.predict(X_test)
# print('y_train_predict', y_train_predict, 'y_test_predict', y_test_predict)
score_train = accuracy_score(y_train, y_train_predict)
score_test = accuracy_score(y_test, y_test_predict)
# score_train 0.9047619047619048 score_test 0.6428571428571429  效果并不好，观察一下分类边界
# print('score_train', score_train, 'score_test', score_test)

# visualize the knn model and boundary
'''
 xx is a 200*200 0-9.95,0-0  yy is a 200*200 0-0,0-9.95
 涉及分类模型决策边界中的np.meshgrid()与np.c_[]两个函数的使用.
 np.meshgrid：会返回两个np.arange类型的列表.是的,列表.
 xx.reval()：将多维列表转换为一维列表
 np.c_[xx.ravel(), yy.ravel()]：按行连接两个一维列表，要求行数相等
exampl:
a = [1, 1, 1]
b = [2, 2, 2]
np.c_[a, b] ===>
[
    [1, 2],
    [1, 2],
    [1, 2]
]   
'''
xx, yy = np.meshgrid(np.arange(0, 10, 0.05), np.arange(0, 10, 0.05))
# print('xx.shape', xx.shape, 'xx', xx)
# print('yy.shape', yy.shape, 'yy.type', type(yy), 'yy', yy)
# print('xx.ravel()', xx.ravel(), 'xx.ravel().type', type(xx.ravel()))
# print('yy.ravel()', yy.ravel())
x_range = np.c_[xx.ravel(), yy.ravel()]
# print('x_range', x_range, 'x_range.shape', x_range.shape)
y_range_predict = KNN_10.predict(x_range)
# visualize the knn_10 predict result
fig4 = plt.figure(figsize=(10, 10))
bad_knn10 = plt.scatter(
    x_range[:, 0][y_range_predict == 0], x_range[:, 1][y_range_predict == 0])
good_knn10 = plt.scatter(
    x_range[:, 0][y_range_predict == 1], x_range[:, 1][y_range_predict == 1])
# 后画原始数据,要在所有数据点上面画出原始数据.若先画原始数据会被覆盖.
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.legend((bad, good, bad_knn10, good_knn10),
           ('bad', 'good', 'bad_knn10', 'good_knn10'))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('knn predict data')
# plt.show()

# confuse matrix
cm = confusion_matrix(y_test, y_test_predict)
print('cm', cm)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
'''
accurary=(TP + TN )/( TP + FP + TN + FN)
灵敏度(sensitivity)/召回率（recall）： TP / (TP + FN)，正确预测为正占全部正样本的比例
特异度(specificity):TN/(TN+FP)  负样本中预测正确的比例
精准率（precision）：TP / (TP + FP)，正确预测为正占全部预测为正的比例
'''
accurary = (TP + TN)/(TP + FP + TN + FN)
recall = TP / (TP + FN)
specificity = TN/(TN+FP)
precision = TP / (TP + FP)
f1 = 2*precision*recall/(precision+recall)


# try different k and calcualte the accuracy for each
n = [i for i in range(1, 21)]
accurary_train = []
accurary_test = []
for i in n:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_train_predict = knn.predict(X_train)
    y_test_predict = knn.predict(X_test)
    score_train = accuracy_score(y_train, y_train_predict)
    score_test = accuracy_score(y_test, y_test_predict)
    accurary_train.append(score_train)
    accurary_test.append(score_test)
print('accurary_train', accurary_train, 'accurary_test', accurary_test)
# visualize the accurary
fig5 = plt.figure(figsize=(10, 5))
plt.plot(n, accurary_train, marker='o')
plt.plot(n, accurary_test, marker='x')
plt.xlabel('n')
plt.ylabel('train or test accurary')
# plt.show()
