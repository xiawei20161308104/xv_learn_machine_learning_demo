'''
采用kmeans算法实现2d数据data.csv聚类
预测（80，60）数据类别
计算准确率矫正结果
之后依次使用knn，meanshift完成
'''
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv('data.csv')
print(data.head())

# define X，y
X = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']
# how many labels:0,1,2
# print(pd.value_counts(y))
# visalize the data
fig1 = plt.figure()
label0 = plt.scatter(X.loc[:, 'V1'][y == 0], X.loc[:, 'V2'][y == 0])
label1 = plt.scatter(X.loc[:, 'V1'][y == 1], X.loc[:, 'V2'][y == 1])
label2 = plt.scatter(X.loc[:, 'V1'][y == 2], X.loc[:, 'V2'][y == 2])
plt.title("lable_data")
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
# plt.show()
# set the model
KM = KMeans(n_clusters=3, random_state=0)
KM.fit(X)
centers = KM.cluster_centers_
fig2 = plt.figure()
label0 = plt.scatter(X.loc[:, 'V1'][y == 0], X.loc[:, 'V2'][y == 0])
label1 = plt.scatter(X.loc[:, 'V1'][y == 1], X.loc[:, 'V2'][y == 1])
label2 = plt.scatter(X.loc[:, 'V1'][y == 2], X.loc[:, 'V2'][y == 2])
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()
# predict v1=80 v2=60
print(KM.predict([[80, 60]]))
y_predict = KM.predict(X)
print(pd.value_counts(y_predict), pd.value_counts(y))
# 很低 0.02
score = accuracy_score(y, y_predict)
# 加一步矫正，很简单，for便利重新赋值
y_corrected = []
for i in y_predict:
    if i == 0:
        y_corrected.append(1)
    elif i == 1:
        y_corrected.append(2)
    else:
        y_corrected.append(0)
score2 = accuracy_score(y, y_corrected)
print('score2=', score2)

# over
