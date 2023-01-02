'''
Version: 1.0
Author: xiawei
Date: 2022-12-27 10:26:51
LastEditors: xiawei
LastEditTime: 2022-12-29 12:20:55
Description: 无前景背景无未知区域,connectedComponentsWithStats
锐化-灰度化-高斯滤波-二值化-开运算-connectedComponentsWithStats-makers+1-wateshed
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
    global gray, binarys
    # 1、锐化
    kernel = np.array([
        [2, 2, 2],
        [2, -16, 2],
        [2, 2, 2]
    ])
    filter2D = cv.filter2D(img, -1, kernel)
    cv.namedWindow('filter2D', cv.WINDOW_NORMAL)
    cv.imshow('filter2D', filter2D)
    # 灰度化
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray', gray)
    # 高斯滤波
    gray = cv.GaussianBlur(gray, (5, 5), 2)
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

    # cv.namedWindow('Markers', cv.WINDOW_NORMAL)
    # cv.imshow('Markers', mark)


# 分水岭找边界
def Watershed():
    global markers
    # 1、开运算去噪
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, (3, 3), iterations=3)
    # cv.imshow('opening', opening)
    """
    etval : 返回值是连通区域的数量。
    labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
    stats ：stats会包含5个参数分别为x,y,h,w,s。分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
    centroids : 返回的是连通区域的质心。
    """
    # ret, markers = cv.connectedComponents(sure_fg)  # 标记最大连通域
    retval, markers, stats, centroids = cv.connectedComponentsWithStats(
        opening, 8)  # 标记最大连通域
    logger.info('connectedComponentsWithStats返回值')
    logger.info(retval)
    logger.info(markers)
    logger.info(stats)
    logger.info(centroids)
    markers = markers+1  # 背景标记为1（此为最大连通域）
    # logger.info('连通markers+1')
    # logger.info(markers)

    # 6、使用分水岭算法，合并不确定区域和种子，边界修改为-1（分界：连通域背景 -- 未知区域+种子）
    markers = cv.watershed(img, markers)  # 分水岭算法（修改边界为-1）
    # Show_Markers()  # 显示各区域（连通域/背景、不确定区域、种子/前景）
    logger.info('分水岭markers')
    logger.info(markers)


if __name__ == '__main__':
    img = cv.imread('./1.png')
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)

    ToBinary()  # 转二值图

    Watershed()  # 分水岭找边界

    cv.waitKey(0)
