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

import dataloaders
import keyboard_model as models
from config import cfg 

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
    
class KeyBoard(object):
    """docstring for KeyBoard"""
    def __init__(self):
        super(KeyBoard, self).__init__()
        self.load_keyboard_model()

    def load_keyboard_model(self):
        config = json.load(open(cfg.KEYBOARD_JSON))
        dataset_type = config['train_loader']['type']
        loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(loader.MEAN, loader.STD)
        self.num_classes = loader.dataset.num_classes
        self.palette = loader.dataset.palette

        self.model = getattr(models, config['arch']['type'])(self.num_classes, **config['arch']['args'])
        availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        checkpoint = torch.load(cfg.KEYBOARD_MODEL)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(model)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def detect_keyboard(self,img):
        with torch.no_grad():
            image = img.convert('RGB')
            input = self.normalize(self.to_tensor(image)).unsqueeze(0)
            prediction = self.model(input.to(self.device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            has_keyboard,keyboard_rect = self.post_process(image,prediction)
            return keyboard_rect,has_keyboard


    def post_process(self,image,mask):
        w, h = image.size
        colorized_mask = colorize_mask(mask, self.palette)
        pmask = np.array(colorized_mask)
        height,width = pmask.shape
        loc_x,loc_y = [],[]
        for i in range(height):
            for j in range(width):
                if pmask[i,j]!=0:
                    loc_y.append(i)
        loc_y.sort()
        loc_y = np.unique(np.array(loc_y))
        locy_min,locy_max = 0,0
        for y in loc_y:
            cmask = np.where(pmask[y]!=0)[0]
            if len(cmask)>0.5*width:
                locy_min = y 
                break 
        for y in loc_y[::-1]:
            cmask = np.where(pmask[y]!=0)[0]
            if len(cmask)>0.5*width:
                locy_max = y 
                break 
        piano_ylen = locy_max-locy_min 
        locx_min,locx_max = 0,0
        for x in range(width):
            cmask = np.where(pmask[locy_min:locy_max,x]!=0)[0]
            if len(cmask)>0.3*(piano_ylen):
                locx_min = x 
                break 
        for x in range(width)[::-1]:
            cmask = np.where(pmask[locy_min:locy_max,x]!=0)[0]
            if len(cmask)>0.3*piano_ylen:
                locx_max = x 
                break
        Rect = (locx_min,locy_min,locx_max,locy_max)
        if locy_max-locy_min<20:
            return False,Rect 
        return True,Rect 
