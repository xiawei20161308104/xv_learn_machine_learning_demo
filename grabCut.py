'''
Version: 1.0
Author: xiawei
Date: 2022-09-06 09:43:26
LastEditors: xiawei
LastEditTime: 2022-12-22 19:04:50
Description: 手动绘制掩码并提取掩码
'''


import cv2
import numpy as np


class App:
    startX = 0
    startY = 0
    flag_rect = False
    rect = (0, 0, 0, 0)

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 标记flag只进行一次画框，不然鼠标移动的时候一直画
            self.flag_rect = True
            # 当时鼠标按下的x和y的值
            self.startX = x
            self.startY = y
            print('x', x, '  y', y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_rect = False
            cv2.rectangle(self.img, (self.startX, self.startX),
                          (x, y), (0, 0, 255), 3)
            self.rect = (min(self.startX, x), min(self.startY, y),
                         abs(self.startX-x), abs(self.startY-y))
        elif event == cv2.EVENT_MOUSEMOVE:
            # 只有按下并且移动才是想画框的时候,255,0,0是红色
            # 其实这里有不理解哦，代码逻辑是移动为红色，实际效果为圈好为红色
            if self.flag_rect == True:
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.startX, self.startX),
                              (x, y), (255, 0, 0), 3)

    def run(self):
        self.img = cv2.imread('1.png')
        self.img2 = self.img.copy()
        self.output = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        cv2.namedWindow('input', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('input', self.onmouse)

        while (1):
            cv2.imshow('input', self.img)
            # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', self.output)
            k = cv2.waitKey(100)
            if k == 27:
                break
            if k == ord('q'):
                # 前景和背景格式
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                # 调用 原始图像，掩码，范围，前景后景，迭代次数，调用格式
                cv2.grabCut(self.img2, self.mask,
                            self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                # 这个掩码需要很大，把前景转换成更明显的格式
                mask2 = np.where((self.mask == 1) | (
                    self.mask == 3), 255, 0).astype('uint8')
                print(mask2.shape)
                self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)


App().run()
