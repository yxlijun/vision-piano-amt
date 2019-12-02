import os 
import argparse 
from models.model_helper import ModelProduct 
import argparse 
from PIL import Image 
import sys 
from config import PROJECT_ROOT,cfg
sys.path.append(os.path.join(PROJECT_ROOT,'piano_utils'))
from keyboard import KeyBoard
import numpy as np 
import cv2 

def str2bool(param):
    return param in ['True','1','true','yes']

parser = argparse.ArgumentParser()
parser.add_argument('--img','-i',type=str,default=None)
parser.add_argument('--color',default='white',choices=['white','black'])
parser.add_argument('--type',default='key',choices=['keyboard','key'])
parser.add_argument('--img_dir','-id',type=str,default=None)
parser.add_argument('--pos',type=str,default='True')
args = parser.parse_args()

save_dir = '/home/data/lj/Piano/saved/other'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def list_img_dir(path,img_lists):
    subpaths = [os.path.join(path,x) for x in os.listdir(path)]
    for subpath in subpaths:
        if os.path.isdir(subpath):
            list_img_dir(subpath,img_lists)
        elif subpath.endswith('jpg'):
            img_lists.append(subpath)

if args.img is not None:
    if args.type=='key':
        modelproduct = ModelProduct()
        input_imgs = []
        img = Image.open(args.img)
        input_imgs.append(img)
        if args.color=='black':
            pred = modelproduct.detect_black_keys(input_imgs,debug=True)
        else: 
            pred = modelproduct.detect_white_keys(input_imgs,debug=True)
        print(pred)
    else:
        keyboard = KeyBoard()
        img = Image.open(args.img)
        keyboard.detect_keyboard(img)
if args.img_dir is not None:
    img_lists = []
    list_img_dir(args.img_dir,img_lists)
    modelproduct = ModelProduct()
    count = 0
    for img_path in img_lists:
        input_imgs = []
        img = Image.open(img_path)
        input_imgs.append(img)
        if args.color=='black':
            pred,prob = modelproduct.detect_black_keys(input_imgs,debug=False)
        else:
            pred,prob = modelproduct.detect_white_keys(input_imgs,debug=False)
        if str2bool(args.pos):
            if pred[0]==1:count+=1
            '''
            else:
                split_line = img_path.split('/')
                save_file = os.path.join(save_dir,'{}_{}'.format(split_line[-2],split_line[-1]))
                img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_file,img)
            '''
        else:
            if pred[0]==0:count+=1
    print('{}/{}={:.3}'.format(count,len(img_lists),count/len(img_lists)))
        


