'''
Version: 1.0
Author: xiawei
Date: 2022-12-24 09:02:25
LastEditors: xiawei
LastEditTime: 2022-12-28 06:59:37
Description: 2
'''
# 距离变换与分水岭算法（路面检测）
import numpy as np
import cv2 as cv


# 转二值图像
def ToBinary():
    global gray, binary
    # 1、锐化
    kernel = np.array([
        [2, 2, 2],
        [2, -16, 2],
        [2, 2, 2]
    ])
    sharp = cv.filter2D(img, -1, kernel)
    # cv.imshow('sharp', sharp)
    # 灰度化
    gray = cv.cvtColor(sharp, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray', gray)
    # 二值化
    ret, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # cv.imshow('binary', binary)


# 显示各区域（连通域/背景、不确定区域、种子/前景）
def Show_Markers():
    mark = img.copy()
    mark[markers == 1] = (255, 0, 0)  # 连通域/背景（蓝）
    mark[markers == 0] = (0, 255, 0)  # 不确定区域（绿）
    mark[markers > 1] = (0, 0, 255)  # 前景/种子（红）

    mark[markers == -1] = (0, 255, 0)  # 边界（绿）
    cv.namedWindow('Markers', cv.WINDOW_NORMAL)
    cv.imshow('Markers', mark)


# 分水岭找边界
def Watershed():
    global markers
    # 1、开运算去噪
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, (3, 3), iterations=3)
    # cv.imshow('opening', opening)

    # 2、确定背景区域（膨胀）
    sure_bg = cv.dilate(opening, (3, 3), iterations=2)
    # cv.imshow('sure_bg', sure_bg)

    # 3、确定前景区域（距离变换）（种子）
    # 3-1、求对象最大宽度/长度（直径）
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3, 5)
    # 3-2、最长直径按比例缩小，确定前景
    ret, sure_fg = cv.threshold(
        dist_transform, 0.01 * dist_transform.max(), 255, cv.THRESH_BINARY)
    #   前景          阈值函数                    阈值                        最大值 二值化方式
    sure_fg = np.uint8(sure_fg)
    # cv.imshow('sure_fg', sure_fg)

    # 4、找未知区域（未知区域 = 确定的背景-确定的前景）
    unknown = cv.subtract(sure_bg, sure_fg)
    # cv.imshow('unknown', unknown)

    # 5、根据种子标记最大连通域（大于1为内部区域，标记1为背景区域，0为未知区域）
    ret, markers = cv.connectedComponents(sure_fg)  # 标记最大连通域
    # retval, markers, stats, centroids = cv.connectedComponentsWithStats(
    #     opening, 8)
    markers = markers+1  # 背景标记为1（此为最大连通域）
    markers[unknown == 255] = 0  # 未知区域标记为0

    Show_Markers()  # 显示各区域（连通域/背景、不确定区域、种子/前景）

    # 6、使用分水岭算法，合并不确定区域和种子，边界修改为-1（分界：连通域背景 -- 未知区域+种子）
    markers = cv.watershed(img, markers)  # 分水岭算法（修改边界为-1）

    Show_Markers()  # 显示各区域（连通域/背景、不确定区域、种子/前景）

    # 7、涂色并显示（边界(markers==-1)涂色）
    dst = img.copy()
    dst[markers == -1] = [0, 255, 0]  # 边界(-1)涂色
    cv.namedWindow('dst', cv.WINDOW_NORMAL)
    cv.imshow('dst', dst)


if __name__ == '__main__':
    img = cv.imread('./14.png')
    # cv.namedWindow('img', cv.WINDOW_NORMAL)
    # cv.imshow('img', img)

    ToBinary()  # 转二值图
    Watershed()  # 分水岭找边界

    cv.waitKey(0)
