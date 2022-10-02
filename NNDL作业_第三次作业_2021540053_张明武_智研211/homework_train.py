from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input,Lambda,Conv2D,Dropout,MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.utils import plot_model
import os
import matplotlib.pyplot as plt

'''训练图像和mask的大小为512*384'''
IMG_WIDTH, IMG_HEIGHT = 256, 256 #设置图片的长和宽信息
X = next(os.walk('./data/images'))[2] #读取文件夹下的图像
y = next(os.walk('./data/masks'))[2]  #读取文件夹下的mask
X_ids = X[:-10] #取数据中前1182个样本
y_ids = y[:-10] #取数据中前1182个样本

X_train = np.zeros((len(X_ids), 256, 256, 3), dtype=np.float32) #创建与图像相同的全零数组
y_train = np.zeros((len(y_ids), 256, 256, 1), dtype=np.bool)    #创建与mask相同的全零数组

#扫描每张图片存入X_train中
for n, id_ in enumerate(X_ids):
    image = load_img(f'./data/images/{id_}', target_size=(IMG_HEIGHT, IMG_WIDTH)) #读取图像,resize图像大小为256*256
    X_train[n] = np.array(image) #存入数组向量中

#扫描每张mask存入y_train中
for n, id_ in enumerate(y_ids):
    image = load_img(f'./data/masks/{id_}', target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale") #设置为grayscale,转换为灰度图
    y_train[n] = np.array(image)[:, :, np.newaxis] #增加一个维度,匹配设置的数组维度

#定义U_net网络结构
def U_net(input_size=(256,256,3)):
    inputs = Input(input_size) #输入图像的大小
    s = Lambda(lambda x: x / 255)(inputs)  #归一化

    #下采样
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #上采样
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = U_net()
model_checkpoint = ModelCheckpoint('unet.h5', monitor='val_accuracy',verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=200, callbacks=[model_checkpoint])
# model.summary()
# plot_model(model,to_file='./figure/unet.jpg')

plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()