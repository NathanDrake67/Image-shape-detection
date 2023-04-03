# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:29:43 2022
Microsoft Windows10 家庭中文版
版本20H2(操作系统内部版本19042.1586)
处理器 lntel(R) Core(TM) i5-8300H CPU @ 2.30GHz2.30 GHz
机带RAM 8.00 GB (7.80 GB可用)
GPU0 lntel(R) UHD Graphics 630
GPU1 NVIDIA GeForce GTX 1050 Ti

主要分为四个步骤
(1)借用sobel算子生成梯度幅值图，梯度方向图
(2)进行非极大值抑制处理
(3)进行双阈值处理，滤波除去孤立噪声点后，以强边缘点为中心搜索连接，舍弃低于低阈值的像素点（置零），保留低阈值~255内的像素点
(4)进行二值化，最终图像的像素点只有0或255两种，保留需要的边缘（车道线）

@author: 1851889-郑博源
"""

import cv2
import numpy as np
import math
#定义进行梯度计算的函数
def grad_img(img):
    global sobel_edge_x
    sobel_edge_x = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)#水平方向梯度图像
    global sobel_edge_y
    sobel_edge_y = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)#竖直方向梯度图像
    global edge
    edge = np.sqrt(np.power(sobel_edge_x, 2) + np.power(sobel_edge_y, 2))#计算梯度图像
    cv2.imshow('grad_img',np.uint8(edge))                                #显示图像
#定义非极大值抑制函数
def no_maximum_suppression_default(dx, dy, edge):
    rows, cols = dx.shape                                                #取图像的长和宽
    gradientDirection = np.zeros(dx.shape)                               #建立和原图像大小相同的空矩阵
    edge_nonMaxSup = np.zeros(dx.shape)                                  #最外层一圈为填充0，不进行处理
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            angle = math.atan2(dy[r][c], dx[r][c])*180/math.pi           #计算梯度方向（梯度角），范围在-180°至180°
            gradientDirection[r][c] = angle                              #将梯度角存入梯度角矩阵
            #将图像分为4个梯度角区间进行分类讨论，区间中心分别为0°，45°，90°，135°直线，即八邻域表示法。
            # 左右方向（-22.5°~22.5°及157.5°~-157.5°）
            if abs(angle) < 22.5 or abs(angle) > 157.5:
                if edge[r][c] > edge[r][c-1] and edge[r][c] > edge[r][c+1]:#沿正负梯度方向，比较当前像素与相邻像素的梯度强度
                    edge_nonMaxSup[r][c] = edge[r][c]                    #若当前像素梯度为最大值，则保留；若不是则置零，余下同理                                   
            # 左上角，右下角方向（22.5°~67.5°及-157.5°~-112.5°）
            if 22.5 <= angle < 67.5 or -157.5 <= angle < -112.5:
                if edge[r][c] > edge[r-1][c-1] and edge[r][c] > edge[r+1][c+1]:
                    edge_nonMaxSup[r][c] = edge[r][c]
            # 上下方向（67.5°~112.5°及-112.5°~-67.5°）
            if 67.5 <= abs(angle) <= 112.5:
                if edge[r][c] > edge[r-1][c] and edge[r][c] > edge[r+1][c]:
                    edge_nonMaxSup[r][c] = edge[r][c]
            # 左下角，右上角方向（112.5°~157.5°及-67.5°~-22.5°）
            if 112.5 < angle <= 157.5 or -67.5 < angle <= -22.5:
                if edge[r][c] > edge[r-1][c+1] and edge[r][c] > edge[r+1][c-1]:
                    edge_nonMaxSup[r][c] = edge[r][c]
    return edge_nonMaxSup

#定义一个检验函数，目的是保证遍历的点在图标范围内，且不包含边缘
#其中r,c是点的坐标，rows,cols则是图片的长、宽
def checkInRange(r, c, rows, cols):
    if 0 <= r < rows and 0 <= c < cols:
        return True                    #若该遍历到的点在图片范围内，则继续，否则舍弃这个点
    else:
        return False

#定义一个遍历循环函数（或曰搜索连接函数，目的是寻找强边缘点周围的弱边缘点，若它们相邻则把他们连成一片
#其中edge_nonMaxSup为经非极大值抑制后的梯度幅值图，edges是储存结果的矩阵,r,c是点的坐标，rows,cols则是图片的长、宽
def trace(edge_nonMaxSup, edge, lowerThresh, r, c, rows, cols):
    if edge[r][c] == 0:                                 
        edge[r][c] = edge_nonMaxSup[r][c]       #将该强边缘点输入结果矩阵
        for i in range(-1, 2):
            for j in range(-1, 2):              #两重循环的目的在于遍历该强边缘点周围的八个像素点，寻找是否有高于最低阈值的像素点
                if checkInRange(r+i, c+j, rows, cols) and edge_nonMaxSup[r+i][c+j] >= lowerThresh:
                    trace(edge_nonMaxSup, edge, lowerThresh, r+i, c+j, rows, cols)    
'''#如果在该强边缘点周围找到了高于最低阈值的像素点，则继续对找到的点周围进行遍历
   从而达到搜索连接强边缘点与若边缘点的目的，于此同时，低于最低阈值的点置零
   孤立的弱边缘点（噪声点）也不做处理，即默认置零，从而达到了保留边缘的效果'''   

#定义了双阈值处理函数，参数分别为：edge_nonMaxSup是经非极大值抑制后的梯度幅值图，lowerThresh, upperThresh分别为最低和最高阈值              
def hysteresisThreshold(edge_nonMaxSup, lowerThresh, upperThresh):
    rows, cols = edge_nonMaxSup.shape             #取图片长和宽值
    edge = np.zeros(edge_nonMaxSup.shape, np.uint8)#边缘置零
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            # 大于高阈值的点确定为边缘点，并以该点为起点进行深度优先搜索
            if edge_nonMaxSup[r][c] >= upperThresh:
                trace(edge_nonMaxSup, edge, lowerThresh, r, c, rows, cols)
            # 小于低阈值的点剔除掉，直接置零
            if edge_nonMaxSup[r][c] < lowerThresh:
                edge[r][c] = 0
    return edge

if __name__ == "__main__":
    #进行预处理：读入图片，并进行高斯滤波。
    img = cv2.imread(r'img/lanes.png', 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    #原始图像的梯度幅值图
    grad_img(img)
    cv2.imwrite("./output_1/1_Gradient_Magnitude_Map.png",np.uint8(edge))
    # 梯度方向非极大值抑制
    edge_nonMaxSup = no_maximum_suppression_default(sobel_edge_x, sobel_edge_y,edge)
    edge_nonMaxSup[edge_nonMaxSup>255] = 255 #将大于255的像素值截断
    cv2.imshow('nms_img',np.uint8(edge_nonMaxSup))
    cv2.imwrite("./output_1/2_No_Maximum_Suppression.png",np.uint8(edge_nonMaxSup))
    # 双阈值后阈值处理
    edge_nonMaxSup = edge_nonMaxSup.astype(np.uint8)
    canny_edge = hysteresisThreshold(edge_nonMaxSup, 60, 180)#选取60和180为上下阈值，尽量保留车道线而减少其他边缘
    cv2.imshow("hysteresisThreshold.png", canny_edge)
    cv2.imwrite("./output_1/3_Hysteresis_Threshold.png",np.uint8(canny_edge))
    #双阈值后处理后的梯度图进行二值化
    canny_edge[canny_edge>0]=255
    cv2.imshow("Binarization",canny_edge)
    cv2.imwrite("./output_1/4_Binarization.png",np.uint8(canny_edge))   
    cv2.waitKey(0)
    cv2.destroyAllWindows()


