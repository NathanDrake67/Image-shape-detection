# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:51:30 2022
Microsoft Windows10 家庭中文版
版本20H2(操作系统内部版本19042.1586)
处理器 lntel(R) Core(TM) i5-8300H CPU @ 2.30GHz2.30 GHz
机带RAM 8.00 GB (7.80 GB可用)
GPU0 lntel(R) UHD Graphics 630
GPU1 NVIDIA GeForce GTX 1050 Ti

采用霍夫梯度法进行霍夫圆的检测，主要利用了梯度方向指向圆心,即该点法线与半径垂直的特性
由任务一中的canny边缘检测，已知各点的梯度方向，各点做梯度方向的直线，经过圆心。
经过的某疑似是圆心的直线越多，此点投票数越多，越有很大概率是圆心
再通过所设定的，圆心间最小距离，最小圆半径，最大圆半径等，筛选出符合要求的圆，是为整体的思路

@author: 1851889-郑博源
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#定义利用霍夫梯度法的霍夫圆检测函数
#参数分别为：输入图像（须是进行过canny边缘检测后的二值图，中心阈值因子（相当于一个放大因子）,投票阈值（高于阈值的圆被选出），最小圆心距离
#最小半径，最大半径，中心轴刻度，半径刻度，半窗口，最大圆数量
def AHTforCircles(edge,center_threshold_factor = None,score_threshold = None,min_center_dist = None,minRad = None,maxRad = None,center_axis_scale = None,radius_scale = None,halfWindow = None,max_circle_num = None):
    min_center_dist_square = min_center_dist**2
    edge_x = cv2.Sobel(edge, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3) #x方向的sobel边缘检测图
    edge_y = cv2.Sobel(edge, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3) #y方向的sobel边缘检测图
    
    center_accumulator = np.zeros((int(np.ceil(center_axis_scale*edge.shape[0])),int(np.ceil(center_axis_scale*edge.shape[1]))))#创建一个计数器（用来存放每个像素点的投票数）,np.ceil()是向上取整函数，center_axis_scale是一个中心缩放因子
    k = np.array([[r for c in range(center_accumulator.shape[1])] for r in range(center_accumulator.shape[0])])
    #创建一个与原图大小相同的矩阵（本此处理图片为147*286），每一行的元素都相同，k矩阵即每一行都是0，1……146，共147行
    l = np.array([[c for c in range(center_accumulator.shape[1])] for r in range(center_accumulator.shape[0])])
    #创建一个与原图大小相同的矩阵（本此处理图片为147*286），每一列的元素都相同，k矩阵即每一列都是0，1……285，共286列
    minRad_square = minRad**2    #最小圆半径的平方，方便后续计算
    maxRad_square = maxRad**2    #最大圆半径的平方，方便后续计算
    points = [[],[]]             #创建一个新的空列表，用于存储各点的横纵坐标

    edge_x_pad = np.pad(edge_x,((1,1),(1,1)),'constant')#对x方向的边缘检测图进行边缘处理，补0补齐
    edge_y_pad = np.pad(edge_y,((1,1),(1,1)),'constant')#对y方向的边缘检测图进行边缘处理，补0补齐
    Gaussian_filter_3 = 1.0 / 16 * np.array([(1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)])#高斯滤波算子

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            if not edge[i,j] == 0:                  #遍历原图中所有非0的像素点，先对边缘检测图进行滤波处理
                dx_neibor = edge_x_pad[i:i+3,j:j+3] #取所选像素周边的八个像素点，形成3×3卷积核，注意此点为卷积核左上角，与高斯算子卷积
                dy_neibor = edge_y_pad[i:i+3,j:j+3] 
                dx = (dx_neibor*Gaussian_filter_3).sum()  #矩阵相乘求和
                dy = (dy_neibor*Gaussian_filter_3).sum()
                if not (dx == 0 and dy == 0):
                    t1 = (k/center_axis_scale-i)   #点所在的那一行为全零行
                    t2 = (l/center_axis_scale-j)   #点所在的那一列为全零列
                    t3 = t1**2 + t2**2             #所筛选的点的条件是：(1)该点到圆心的距离在最大最小半径范围内；(2)该点与圆心的连线与该点梯度方向共线（即相当于半径垂直与法线判断是否在圆上）
                    temp = (t3 > minRad_square)&(t3 < maxRad_square)&(np.abs(dx*t1-dy*t2) < 1e-4)
                    center_accumulator[temp] += 1 #符合要求则将点加入数据集
                    points[0].append(i)          #存入点的纵坐标
                    points[1].append(j)          #存入点的横坐标

    M = center_accumulator.mean()   #M是整个矩阵的平均值，相当于成为了一个判定是否能成为圆心的比例因子
    for i in range(center_accumulator.shape[0]):
        for j in range(center_accumulator.shape[1]):  #圆心计数器
            neibor = \
                center_accumulator[max(0, i - halfWindow + 1):min(i + halfWindow, center_accumulator.shape[0]),
                max(0, j - halfWindow + 1):min(j + halfWindow, center_accumulator.shape[1])]
            if not (center_accumulator[i,j] >= neibor).all():
                center_accumulator[i,j] = 0
            # 非极大值抑制：只保留梯度最大值的方向
    #对投票矩阵进行整型（像素转化至0~255，以线性差值的方式进行）
    votemap = cv2.normalize(center_accumulator, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Voting map",votemap)                            #投票热图的显示与存储
    cv2.imwrite("./output_2/2_Voting_map_Wheel.png",votemap)


    center_threshold = M * center_threshold_factor
    possible_centers = np.array(np.where(center_accumulator > center_threshold))  
    # 挑选可能是圆心的点，判断标准是该点的实际投票值大于该点的可能是圆心的阈值


    sort_centers = []
    for i in range(possible_centers.shape[1]):
        sort_centers.append([])                           #建立存储圆的列表
        sort_centers[-1].append(possible_centers[0,i])    #对应每一个圆，输入它的横纵坐标，及投票数
        sort_centers[-1].append(possible_centers[1,i])
        sort_centers[-1].append(center_accumulator[sort_centers[-1][0],sort_centers[-1][1]])

    sort_centers.sort(key=lambda x:x[2],reverse=True) #将矩阵内的各项按每项第一个元素（投票数）大小进行升序排列，投票数最大的在第一位

    centers = [[],[],[]]
    points = np.array(points)
    for i in range(len(sort_centers)):
        radius_accumulator = np.zeros(
            (int(np.ceil(radius_scale * min(maxRad, np.sqrt(edge.shape[0] ** 2 + edge.shape[1] ** 2)) + 1))),dtype=np.float32)
        if not len(centers[0]) < max_circle_num:
            break
        iscenter = True
        #比较现有圆是否距其他已经筛选出的圆心距离太近
        for j in range(len(centers[0])): 
            d1 = sort_centers[i][0]/center_axis_scale - centers[0][j]   #圆心距在x轴投影
            d2 = sort_centers[i][1]/center_axis_scale - centers[1][j]   #圆心距在y轴投影
            if d1**2 + d2**2 < min_center_dist_square:                  #若现有圆心距小于最小圆心距离，舍弃这个点
                iscenter = False
                break

        if not iscenter:
            continue

        temp = np.sqrt((points[0,:] - sort_centers[i][0] / center_axis_scale) ** 2 + (points[1,:] - sort_centers[i][1] / center_axis_scale) ** 2) #求解点和圆心之间的距离
        temp2 = (temp > minRad) & (temp < maxRad)         #假如圆的半径符合所设置的最大最小半径值，则进行下一步处理
        temp = (np.round(radius_scale * temp)).astype(np.int32)  #向上取整，转为32位int类型
        for j in range(temp.shape[0]):
            if temp2[j]:
                radius_accumulator[temp[j]] += 1                #符合要求的点添加至数据集
        for j in range(radius_accumulator.shape[0]):
            if j == 0 or j == 1:
                continue
            if not radius_accumulator[j] == 0:
                radius_accumulator[j] = radius_accumulator[j]*radius_scale/np.log(j) #进行类似归一化，像素值在合理范围内
        score_i = radius_accumulator.argmax(axis=-1)
        if radius_accumulator[score_i] < score_threshold:           #若获得投票数小于投票阈值，则舍弃这个点
            iscenter = False

        if iscenter:
            centers[0].append(sort_centers[i][0]/center_axis_scale) #第一部分存储圆心的横坐标
            centers[1].append(sort_centers[i][1]/center_axis_scale) #第二部分存储圆心的纵坐标
            centers[2].append(score_i/radius_scale)                 #第三部分存储圆的半径


    centers = np.array(centers)                                     #矩阵化
    centers = centers.astype(np.float64)                            #整型

    return centers


#定义一个画圆函数，输入参数为筛选出的圆，原图的canny检测图，线条的颜色（R,G,B）,线条粗细度
def drawCircles(circles,edge,color = (0,0,255),err = 80):
    result = edge
    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            dist_square = (circles[0]-i)**2 + (circles[1]-j)**2 #得到圆心到某像素点的距离的平方
            e = np.abs(circles[2]**2 - dist_square)             #e值是该圆半径的平方到上述距离的平方的差值，检测该像素是否在圆内。
            if (e < err).any():
                result[i,j] = color                             #圆心是蓝色
            if (dist_square < 25.0).any():                      #圆圈是红色
                result[i,j] = (255,0,0)
    return result


if __name__=='__main__':

    img = cv2.imread(r'img/wheel.png')           #读入图像
    blurred = cv2.GaussianBlur(img, (3, 3), 0)   #对图像进行高斯滤波处理
    cv2.imshow('origin',blurred)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)#图像灰度化
    gray = gray.astype( np.uint8 )                #图像转为能够处理的uint8类型（原型为float32)
    edge = cv2.Canny(gray, 50, 150)               #得到canny边缘检测的二值图，像素点为 0 或 255
    cv2.imshow("canny_edge",edge)                 #输出并保存边缘检测的结果
    cv2.imwrite("./output_2/1_Canny_edge_Wheel.png",edge)
    circles = AHTforCircles(edge,3.33,18,100,10,20,1.0,1,2,2)       
    '''最优参数调整如下：
        center_threshold_factor': 3.33,
        'score_threshold':18.0,
         'min_center_dist':100.0,
         'minRad':10,
         'maxRad':20,
         'center_axis_scale':1.0,
         'radius_scale':1.0,
         'halfWindow':2,
        'max_circle_num':2'''
        
    final_img = drawCircles(circles,blurred)
    cv2.imshow("Hough circle",final_img)
    cv2.imwrite("./output_2/3_Hough_circle_Wheel.png",final_img)

