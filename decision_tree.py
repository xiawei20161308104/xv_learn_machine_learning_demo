"""
基于iris_data.csv(燕尾花数据集)建立决策树模型，评估模型表现
可视化决策树结构
修改leaf参数，对比模型结果
"""
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
data = pd.read_csv('iris_data.csv')
print(data.head())
X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
print(X.shape, y.shape)
# 参数为叶子节点最少样本数，修改之后树的分支深度不同
dc_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)
dc_tree.fit(X, y)

score = accuracy_score(y, dc_tree.predict(X))
print('dc_tree_score', score)

# 可视化模型结构
fig = plt.figure(figsize=(10, 10))
tree.plot_tree(dc_tree, filled='True', feature_names=[
               'a', 'b', 'c', 'd'], class_names=['11', '22', '33'])
plt.show()
