"""
1.基于chip_test.csv数据，建立二阶边界的逻辑回归模型
2.以函数的方式求解边界曲线
3.描绘出完整的边界曲线
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load data
data = pd.read_csv('chip_test.csv')
# print(pd.DataFrame(data.head()))
# visualize the data
fig1 = plt.figure()
mask = data.loc[:, 'pass'] == 1
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
plt.legend((passed, failed), ('passed', 'failed'))
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.show()

# define data
X = data.drop(['pass'], axis=1)
y = data.loc[:, 'pass']
X1 = data.loc[:, 'test1']
X2 = data.loc[:, 'test2']
X1.head()
# create new data
X1_2 = X1*X1
X2_2 = X2*X2
X1_X2 = X1*X2
# 此时创造了新的x变量，将新的x组装传入模型训练，y是不变的
X_new = {'X1': X1, 'X2': X2, 'X1_2': X1_2, 'X2_2': X2_2, 'X1_X2': X1_X2}
X_new = pd.DataFrame(X_new)
print(X_new.head())
# create model and train and predict and score
LR2 = LogisticRegression()
LR2.fit(X_new, y)
score2 = accuracy_score(y, LR2.predict(X_new))
print(score2)

# define fx,以函数方式求解边界曲线


def f(x):
    return 1, 2
# 烂尾。后面就是调用fx函数求对应的x2
