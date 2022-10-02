"""
基于anomaly_data.csv数据，可视化数据分布
可视化对应的高斯分布的概率密度函数
建立模型 实现异常数据点检测
可视化异常检测的处理结果
修改概率分布阈值，查看影响  
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
# 协方差，异常检测包
from sklearn.covariance import EllipticEnvelope
data = pd.read_csv('anomaly_data.csv')
# print(data.head())
fig = plt.figure(figsize=(10, 5))
plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'])
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()


x1 = data.loc[:, 'x1']
x2 = data.loc[:, 'x2']
fig2 = plt.figure(figsize=(20, 5))
plt.subplot(121)
# 直方图是用来衡量连续变量的概率分布的。
# 在构建直方图之前，我们需要先定义好bin（把值分成多少等份），
# 也就是说我们需要先把连续值划分成不同等份，然后计算每一份里面数据的数量
plt.hist(x1, bins=100)
plt.title('x1 distribution')
plt.xlabel('x1')
plt.ylabel('number')

plt.subplot(122)
plt.hist(x1, bins=100)
plt.title('x2 distribution')
plt.xlabel('x2')
plt.ylabel('number')
plt.show()

# calculate the mean and sigma of x1,x2 计算均值和标准差
x1_mean = x1.mean()
x1_sigma = x1.std()

x2_mean = x2.mean()
x2_sigma = x2.std()

# calculate the gassusian distribution p(x) 高斯分布的概率密度
# 创建一个横坐标范围，均匀的密度点
x1_range = np.linspace(0, 20, 300)
# norm函数求解概率密度,坐标范围,均值和方差
x1_normal = norm.pdf(x1_range, x1_mean, x1_sigma)
x2_normal = norm.pdf(x1_range, x2_mean, x2_sigma)
# visualize the px
fig = plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.plot(x1_range, x1_normal)
plt.title("px1")
plt.subplot(122)
plt.plot(x1_range, x2_normal)
plt.title("px2")
# plt.show()
# 以上结束了对数据进行高斯可视化
# 以下要建立模型,实现异常检测,可视化异常检测结果
ad_model = EllipticEnvelope()
ad_model.fit(data)
y_predict = ad_model.predict(data)
# 1有276  -1有31  -1即为异常点
print(pd.value_counts(y_predict))

fig5 = plt.figure(figsize=(10, 5))
orginal_data1 = plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'], marker='x')
anomaly_data2 = plt.scatter(data.loc[:, 'x1'][y_predict == -1],
                            data.loc[:, 'x2'][y_predict == -1], marker='o')
# ,facecolor = 'none', edgecolor = 'red', s = 150
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((orginal_data1, anomaly_data2), ('orginal_data', 'anomaly_data'))
plt.show()


# 修改模型阈值参数，检测效果不同，
# 模型 contamination默认为0.1  修改为0.02，检测条件变松。
