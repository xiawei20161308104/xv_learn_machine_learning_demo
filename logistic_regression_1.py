
'''
逻辑回归demo
1.基于examdata.csv数据建立回归模型，预测（75，60）的数据下，第三次是否通过
2.对以上模型建立二阶边界，提高精准度
3.画二阶函数图
'''
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load data
data = pd.read_csv('examdata.csv')
# print(pd.DataFrame(data.head()))
# visualize the data
fig1 = plt.figure()
'''
# Python中通过索引名称提取数据,loc()函数。Python中通过行和列下标提取数据，iloc()函数。
# loc括号里面是先行后列，以逗号分割，行标签和行标签之间，列标签和列标签之间用冒号隔开。前闭后闭
# 行列数选择数据，前闭后开
loc和iloc demo
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [
                    7, 8, 9]}, index=["a", "b", "c"])
print(data.loc['b':'c', 'B':'C'])
print(data.iloc[1:3, 1:3])
'''
plt.scatter(data.loc[:, 'Exam1'], data.loc[:, 'Exam2'])
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
# plt.show()
mask = data.loc[:, 'Pass'] == 1
# print(mask)
fig2 = plt.figure()
# 在这里标记了掩码为true的数据，所以只展示通过考试的数据
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
plt.title('Pass_Data')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
# 将两种数据做标记展示
plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()
# dafine X y
# 去除Pass，1表示列
X = data.drop(['Pass'], axis=1)
y = data.loc[:, 'Pass']
X1 = data.loc[:, 'Exam1']
X2 = data.loc[:, 'Exam2']
print(X.head())
# establish model and train it
LR = LogisticRegression()
LR.fit(X, y)
# predicted
y_predicted = LR.predict(X)
# score
score = accuracy_score(y, y_predicted)
# 结果为0.89
print(score)
y_test = LR.predict([[70, 65]])
# print('pass' if y_test == 1 else 'failed')
# 截距，系数1，2.  边界曲线为：theta0+theta1*x1+theta2*x2=0
theta0 = LR.intercept_
theta1, theta2 = LR.coef_[0][0], LR.coef_[0][1]
# print(theta0, theta1, theta2)
# 此时x2可以用x1表示，可以在plt中画出x1和x2点组成的直线，在画出散点图，就是带有逻辑回归分界线的散点图。

# 一阶边界函数不够准确，结果为0.89，所以构造二阶边界函数，就是曲线，会更加适配。
# theta0+theta1*X1+theta2*X2+theta3*X1^2+theta4*X2^2+theta5X1*X2=0
X1_2 = X1*X1
X2_2 = X2*X2
X1_X2 = X1*X2
# 此时创造了新的x变量，将新的x组装传入模型训练，y是不变的
X_new = {'X1': X1, 'X2': X2, 'X1_2': X1_2, 'X2_2': X2_2, 'X1_X2': X1_X2}
X_new = pd.DataFrame(X_new)
# print(X_new)
LR2 = LogisticRegression()
LR2.fit(X_new, y)
score2 = accuracy_score(y, LR2.predict(X_new))
# 改进边界函数之后，分数为1，全部预测成功
print(score2)
# 此时模型里携带边界曲线的参数，theta0为intercept，theta12345为coef[0][0][0][1][0][2]
# 将边界曲线公式转为一元二次函数，x1视为常量。得到x1 x2关于theta的表达式
# 进行可视化的时候需要对x1坐标进行排序，以保证画图的时候是按照顺序点画的图
