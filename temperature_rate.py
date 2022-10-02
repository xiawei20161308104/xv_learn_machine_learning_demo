"""
基于T-R-train.csv数据建立回归模型
对T-R-test.csv数据上的r2分数，可视化模型预测的结果

加入2/5次多项式，建立回归模型
计算r2分数，判断哪个模型更准确
可视化模型预测结果，判断哪个模型更准确
"""
# 生成多项式系数
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# load the data
data_train = pd.read_csv('T-R-train.csv')
# print('data_train', data_train)

# define X_train and y_train
X_train = data_train.loc[:, 'T']
y_train = data_train.loc[:, 'rate']
# print('X_train', X_train, 'y_train', y_train)

# visualize the data
fig0 = plt.figure(figsize=(5, 5))
plt.scatter(X_train, y_train)
plt.title('data_train')
plt.xlabel('T')
plt.ylabel('rate')
# plt.show()

# linear regression model prediction
LR1 = LinearRegression()
X_train = np.array(X_train).reshape(-1, 1)
LR1.fit(X_train, y_train)
y_train_predict = LR1.predict(X_train)
y_train = np.array(y_train)
# print('y_train_predict', y_train_predict)
# print('y_train', y_train)
# TODO:这句报错 不知道为什么，应该是线性回归不能用accurary——acore
# score_train_LR1 = accuracy_score(y_train, y_train_predict)
# mse接近0 r2接近1
mse_train_LR1 = mean_squared_error(y_train, y_train_predict)
r2_train_LR1 = r2_score(y_train, y_train_predict)
# mse_train_LR1 0.030232066637163553 r2_train_LR1 0.016665703886981964  明显，很不好
# print("mse_train_LR1", mse_train_LR1, 'r2_train_LR1', r2_train_LR1)


# 在测试集上运行LR1模型
# loda the test data
data_test = pd.read_csv('T-R-test.csv')
# print('data_test', data_test)

# define X_test and y_test
X_test = data_test.loc[:, 'T']
y_test = data_test.loc[:, 'rate']
# visualize the data
fig1 = plt.figure(figsize=(5, 5))
plt.scatter(X_test, y_test)
plt.title('data_test')
plt.xlabel('T')
plt.ylabel('rate')
# plt.show()
X_test = np.array(X_test).reshape(-1, 1)
# make test data predict on linear regression model
predict_y_test = LR1.predict(X_test)
r2_score_test = r2_score(y_test, predict_y_test)
# print('r2_score_test', r2_score_test)


# visualize the model
# generate the data
# 从40-90之中生成300条数据,并将生成的array转为300行1列的array
X_range = np.linspace(40, 90, 300).reshape(-1, 1)
y_range_LR1_predict = LR1.predict(X_range)
fig2 = plt.figure(figsize=(5, 5))
plt.scatter(X_range, y_range_LR1_predict)
plt.scatter(X_train, y_train)
plt.title('LR1 model predict')
plt.xlabel('T')
plt.ylabel('rate')
# plt.show()


# 多项式模型
# generate polynomialfeatures 生成多项式特征系数
# 生成二次多项式 degree=2
# 当生成的五维多项式的时候，训练数据集r2分数很高接近1，但是测试集只有0.5左右
poly2 = PolynomialFeatures(degree=2)
X_2_train = poly2.fit_transform(X_train)
# 第一次训练之后第二次就不用在训练了
X_2_test = poly2.transform(X_test)
# 有三列，x之前是一列，现在变为了x=a+bx+cx2
# print(X_2_train)
# fit model，predict，score
LR2 = LinearRegression()
LR2.fit(X_2_train, y_train)
y_2_train_predict = LR2.predict(X_2_train)
y_train = np.array(y_train)
mse_train_LR2 = mean_squared_error(y_train, y_2_train_predict)
r2_train_LR2 = r2_score(y_train, y_2_train_predict)
# print(r2_train_LR2)

# visualize the model
# 画横坐标随机点，用ploy对随机点转成二次式得到新的x2（平方），输入模型做预测，并将得到的y画出。
X_2_range = np.linspace(40, 90, 300).reshape(-1, 1)
X_2_range = poly2.transform(X_2_range)
y_2_range_predict = LR2.predict(X_2_range)

fig3 = plt.figure(figsize=(10, 5))
plt.plot(X_range, y_2_range_predict)
plt.scatter(X_train, y_train)
plt.show()
