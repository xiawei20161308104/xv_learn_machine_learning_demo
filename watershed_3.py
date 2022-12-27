'''
Version: 1.0
Author: xiawei
Date: 2022-12-27 10:26:51
LastEditors: xiawei
LastEditTime: 2022-12-27 22:06:39
Description: 双边滤波
锐化-灰度化-双边滤波-二值化-开闭运算-得到未知-connectedComponentsWithStats-makers+1-未知置0-wateshed

'''
# 距离变换与分水岭算法
from logger import logger
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
logger.info('ss')
np.set_printoptions(edgeitems=10, linewidth=160)

# 转二值图像


def ToBinary():
    global gray, binary
    # 1、锐化
    kernel = np.array([
        [10, 10, 10],
        [10, -80, 10],
        [10, 10, 10]
    ])
    sharp = cv.filter2D(img, -1, kernel)
    # cv.imshow('sharp', sharp)
    # 灰度化
    gray = cv.cvtColor(sharp, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray', gray)
    # 均值/高斯/双边滤波
    # gray = cv.medianBlur(gray, 3)
    # gray = cv.GaussianBlur(gray, (5, 5), 2)
    gray = cv.bilateralFilter(gray, 10, 13, 23, 10)
    # 二值化
    ret, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    logger.info('binnary')
    logger.info('binary')
    # cv.namedWindow('binary', cv.WINDOW_NORMAL)
    # cv.imshow('binary', binary)


# 显示各区域（连通域/背景、不确定区域、种子/前景）
def Show_Markers():
    """
     [0, 0, 255]红色
     [255, 0, 0]蓝色
     [0, 255, 0]绿色
     [255, 255, 255]白色
     [255, 255, 0]青色
     [0, 255, 255]黄色
    """
    mark = img.copy()
    mark[markers == 1] = (255, 0, 0)  # 连通域/背景（蓝）
    # mark[markers == 0] = (0, 255, 0)  # 不确定区域（绿）
    # mark[markers > 1] = (0, 0, 255)  # 前景/种子（红）

    mark[markers == -1] = (0, 255, 0)  # 边界（绿）

    cv.namedWindow('Markers', cv.WINDOW_NORMAL)
    cv.imshow('Markers', mark)


# 分水岭找边界
def Watershed():
    global markers

    # 1、开运算去噪
    close = cv.morphologyEx(
        binary, cv.MORPH_CLOSE, (3, 3), iterations=3)
    # cv.namedWindow('opening', cv.WINDOW_NORMAL)
    # cv.imshow('opening', opening)
    dist_transform = cv.distanceTransform(close, cv.DIST_L2, 3, 5)
    cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
    # 3-2、最长直径按比例缩小，确定前景
    ret, sure_fg = cv.threshold(
        dist_transform, 0.01 * dist_transform.max(), 255, cv.THRESH_BINARY)
    #   前景          阈值函数                    阈值                        最大值 二值化方式
    sure_fg = np.uint8(sure_fg)
    # 4、找未知区域（未知区域 = 确定的背景-确定的前景）
    unknown = cv.subtract(sure_fg, close)
    """
    etval : 返回值是连通区域的数量。
    labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
    stats ：stats会包含5个参数分别为x,y,h,w,s。分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
    centroids : 返回的是连通区域的质心。
    """
    # ret, markers = cv.connectedComponents(sure_fg)  # 标记最大连通域
    retval, markers, stats, centroids = cv.connectedComponentsWithStats(
        sure_fg, 4)  # 标记最大连通域
    logger.info('connectedComponentsWithStats返回值')
    logger.info(retval)
    logger.info(markers)
    logger.info(stats)
    logger.info(centroids)
    markers = markers+1  # 背景标记为1（此为最大连通域）
    markers[unknown == 255] = 0  # 未知区域标记为0
    logger.info('分水岭之前markers')
    logger.info(markers)

    # 6、使用分水岭算法，合并不确定区域和种子，边界修改为-1（分界：连通域背景 -- 未知区域+种子）
    markers = cv.watershed(img, markers)  # 分水岭算法（修改边界为-1）
    Show_Markers()  # 显示各区域（连通域/背景、不确定区域、种子/前景）
    logger.info('分水岭markers')
    logger.info(markers)


if __name__ == '__main__':
    img = cv.imread('./1.png')
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)

    ToBinary()  # 转二值图

    Watershed()  # 分水岭找边界

    cv.waitKey(0)
