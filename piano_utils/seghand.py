import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import sys 
sys.path.insert(0,os.path.abspath(__file__))
import cv2
from networks import PSPNet
from config import cfg 
from util import colorize_mask

    
class SegHand(object):
    """docstring for KeyBoard"""
    def __init__(self):
        super(SegHand, self).__init__()
        self.load_handseg_model()
        print('->>finish seg hand model load')

    def load_handseg_model(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(cfg.MEAN, cfg.STD)
        self.num_classes = cfg.HAND_SEG_NUM_CLASSES
        self.palette = cfg.HAND_SEG_PALETTE

        self.model = PSPNet(num_classes=self.num_classes)
        availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        checkpoint = torch.load(cfg.HAND_SEG_MODEL)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(model)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()


    def segment_detect_hand(self,img,rect):
        with torch.no_grad():
            width,height = img.size 
            crop_img = img.crop((0,rect[1],width,height))
            image = crop_img.convert('RGB')
            input = self.normalize(self.to_tensor(image)).unsqueeze(0)
            prediction = self.model(input.to(self.device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            self.hand_box,mask = self.post_process(image,prediction,rect)
            padmask = np.zeros((rect[1],width))
            self.mask = np.concatenate((padmask,mask),axis=0)
            return self.hand_box,self.mask

    def post_process(self,image,mask,rect):
        colorized_mask = colorize_mask(mask, self.palette)
        pmask = np.array(colorized_mask)
        contours, hier = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_box = []
        for cidx,cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            if h>50:
                left_up,right_bottom = (int(x),int(y+rect[1])),(int(x+w),int(y+h+rect[1]))
                hand_box.append((left_up,right_bottom))
        return hand_box,pmask 
        
