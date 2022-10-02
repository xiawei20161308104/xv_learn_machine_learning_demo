from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
from PIL import Image
import random
import os

IMG_WIDTH, IMG_HEIGHT = 256, 256 #设置图片的长和宽信息
X = next(os.walk('./data/test'))[2] #读取文件夹下的图像
test_id = random.choice(X[:])
print(test_id)

img = load_img(f"./data/test/{test_id}", target_size=(IMG_HEIGHT, IMG_WIDTH))
input_array = np.array([img_to_array(img)]) # Convert single image to a batch.
model = load_model('unet_16_200_0.99563.h5') 
pre = model.predict(input_array)
print(pre)

Image.open(f"./data/test/{test_id}").resize((256, 256))
plt.imshow(array_to_img(np.squeeze(pre)[:, :, np.newaxis]))