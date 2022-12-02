'''
Author: xv rg16xw@163.com
Date: 2022-09-17 13:54:36
LastEditors: xv rg16xw@163.com
LastEditTime: 2022-11-24 09:04:03
FilePath: \xv_learn_machine_learning_demo\mnistDistinguish.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
mlp
构建一个普通的两层神经网络对minist手写数字集进行训练
这次只是对流程的概述，具体细节会逐步展开
"""
# In[]
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
mnist = tf.keras.datasets.mnist

# 数据位置，自己运行时候需要重新设置自己的路径
root_dir = ''
data_name = 'mnist.npz'
data_path = root_dir + data_name

# 如果没有数据路径参数，直接从网上下载，有时候会下载比较慢
is_down = True
if is_down:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = mnist.load_data(data_path)

# In[]
# 查看数据shape
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# In[]
# 对图像数据归一化处理
x_train, x_test = x_train / 255.0, x_test / 255.0

# In[]
# 建立神经网络模型
# Flatten()是把28*28的二维数据转为一维向量
# dropout 控制过拟合的方式 Dense 多少个神经元进行全连接
# 第二个dense是因为十个数字所以最后是十个神经元，用softmax一个判断概率的激活函数
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# In[]
# 设置神经网络训练参数，对于优化方法，初学者可以直接选择adam，
# 它是自适应的，可以直接使用缺省参数，大多数情况下，优化效果都很好。
"""
如果你的 targets 是 one-hot 编码，用 categorical_crossentropy
　　one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
 
如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
　　数字编码：2, 0, 1
"""
# In[]
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[]
# 进行训练， batch_size=32是缺省值，当然也可以自己设置，verbose = 1，缺省值，控制进度输出，可以取0，1，2
# 一个epoch表示所有训练样本训练完一次， 一个epoch训练次数 = 训练样本总数/batch_size
model.fit(x_train, y_train, verbose=2, epochs=10, batch_size=100)


# 训练完后，用测试数据进行评估
score = model.evaluate(x_test, y_test)
print(score)


# 评估新图像
new_pic = load_img('./test.jpg', target_size=(28, 28))
new_pic = img_to_array(new_pic)/255
new_pic = new_pic.reshape(1, 28, 28, 3)
result = model.predict(new_pic, verbose=1)
print(np.argmax(result, axis=1))
