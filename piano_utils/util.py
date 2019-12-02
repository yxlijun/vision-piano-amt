#-*- coding:utf-8 -*-
import numpy as np 
import cv2
from PIL import Image
import torch 
import math 

class Stack(object):
    #----list方式实现栈
    def __init__(self):
        self.items = []
    def is_empty(self):
        return self.items == []
    def peek(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def find_connect_domain(img):
    h,w = img.shape[:2]
    dst = img.copy()
    rightBoundary,topBoundary,bottomBoundary,labelValue = 0,0,0,0
    pointstack = Stack()
    boxes = []
    for i in range(h):
        for j in range(w):
            if dst[i,j]!=0:
                continue 
            area = 0 
            labelValue+=1
            seed = (i,j)
            dst[seed] = labelValue
            pointstack.push(seed)
            area+=1
            leftBoundary = seed[1]
            rightBoundary = seed[1]
            topBoundary = seed[0]
            bottomBoundary = seed[0]   #---这个和C++版本的有区别
            while (not (pointstack.is_empty())):
                neighbor = (seed[0], seed[1] + 1)  #---right
                if (seed[1] != (w - 1)) and (dst[neighbor] == 0):
                    dst[neighbor] = labelValue
                    pointstack.push(neighbor)
                    area += 1
                    if (rightBoundary < neighbor[1]):
                        rightBoundary = neighbor[1]
                neighbor = (seed[0]+1, seed[1])  #---bottom
                if ((seed[0] != (h - 1)) and (dst[neighbor] == 0)):
                    dst[neighbor] = labelValue
                    pointstack.push(neighbor)
                    area += 1
                    if (bottomBoundary < neighbor[0]):
                        bottomBoundary = neighbor[0]
                neighbor = (seed[0], seed[1]-1)  #---left
                if ((seed[1] != 0) and (dst[neighbor] == 0)):
                    dst[neighbor] = labelValue
                    pointstack.push(neighbor)
                    area += 1
                    if (leftBoundary > neighbor[1]):
                        leftBoundary = neighbor[1]  
                neighbor = (seed[0]-1, seed[1])   #---top
                if ((seed[0] != 0) and (dst[neighbor] == 0)):
                    dst[neighbor] = labelValue
                    pointstack.push(neighbor)
                    area += 1
                    if (topBoundary > neighbor[0]):
                        topBoundary = neighbor[0]
                seed = pointstack.peek()  #--取栈顶元素并出栈
                pointstack.pop()
            box = (leftBoundary, topBoundary, rightBoundary - leftBoundary, bottomBoundary - topBoundary)  #--(x,y,w,h)
            if area>500:    #---排除一些小框框,二值化的时候表现的不太好
                boxes.append(box)
        return boxes 

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def calAngle( x1,  y1,  x2,  y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan(dy/dx)
    return (angle * 180 / math.pi)

