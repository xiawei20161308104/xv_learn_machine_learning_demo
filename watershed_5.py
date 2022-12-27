'''
Version: 1.0
Author: xiawei
Date: 2022-12-27 21:10:10
LastEditors: xiawei
LastEditTime: 2022-12-27 21:26:39
Description: 在实现1.png基础上改进参数不适用等问题
'''

# 读取,灰度,高斯,二值化
import numpy as np
import cv2 as cv2
img = cv2.imread('./4.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)
# blurred = cv2.bilateralFilter(gray_img, 3, 13, 23, 10)
threshold = cv2.threshold(blurred, 200, 255,
                          cv2.THRESH_BINARY_INV)[1]
# 对二值化分析连通域
totalLabels, label_ids, stats, centroid = cv2.connectedComponentsWithStats(threshold,
                                                                           4,
                                                                           cv2.CV_32S)
# 定义输出矩阵
output = np.zeros(gray_img.shape, dtype="uint8")
print('totalLabels', totalLabels)
for i in range(1, totalLabels):

    # 提取当前标签的连通分量统计信息,这里提取面积
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroid[i]
    print('w', w)
    print('h', h)
    print('area', area)
    # 确保宽高以及面积既不太大也不太小
    keepWidth = w > 3000 and w < 3665
    keepHeight = h > 2000 and h < 2750
    keepArea = area > 3320 and area < 9460540
    # keepWidth = w == 3664
    # keepHeight = h == 2748
    # keepArea = area == 9471470
    # 随着高斯核变大 区域面积变大9512393,因为模糊了——9550580
    if (area > 5000000):
        print('come in')
        componentMask = (label_ids == i).astype("uint8") * 255
        output = cv2.bitwise_or(output, componentMask)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', output)
    """if all((keepWidth, keepHeight, keepArea)):

        print('21')
        componentMask = (label_ids == i).astype("uint8") * 255
        output = cv2.bitwise_or(output, componentMask)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', output)"""

img22 = img.copy()
canny_img = cv2.Canny(output, 150, 300)

# 查找轮廓api，原始图像，查找轮廓方式RETR_EXTERNAL为之查找外围，List从里到外从右到左，CCOMP有层级关系没搞懂
# 从大到小从右到左,这里的contours为根据CHAIN_APPROX_SIMPLE设定的点。
# 返回轮廓个数和坐标值
area2 = []
contours, _ = cv2.findContours(
    canny_img, cv2.TERM_CRITERIA_MAX_ITER, cv2.CHAIN_APPROX_SIMPLE)
for k in range(len(contours)):
    area_k = cv2.contourArea(contours[k])
    if (area_k > 6000):
        print('come in 222')
        area2.append(k)
# max_idx = np.argmax(np.array(area2))
# 最大像素为212129.5
# print("area2222", cv2.contourArea(contours[max_idx]))
# 绘制轮廓，-1为绘制所有轮廓，颜色，线宽。
# print('masker', _)
# all_img2 = cv2.drawContours(img, contours, -1, (0, 0, 255), 5)
# cv2.namedWindow('all_img2', cv2.WINDOW_NORMAL)
# cv2.imshow('all_img2', all_img2)
for j in area2:
    img2 = cv2.drawContours(img, contours, j, (0, 255, 0), 5)
    output = cv2.bitwise_or(img, img2)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', output)

# cv2.namedWindow('canny_img', cv2.WINDOW_NORMAL)
# cv2.imshow('canny_img', canny_img)
cv2.waitKey()


cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img22)
cv2.waitKey(0)
