import torch
import torch.nn as nn 
import argparse 
import os 
from models.resnet18 import ResNet18
import torchvision.transforms as transforms 
from PIL import Image 
import time
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir','-i',type=str,default='.',help='test image')
parser.add_argument('--model','-m',type=str,help='test model')
parser.add_argument('--base_channel','-b',type=int,default=16,help='model base channel')
args = parser.parse_args()

def main(img_dir,model,transform,device):
    img_list = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if x.endswith('jpg')]
    input_imgs = []
    stime = time.time()
    for img_file in img_list:
        img = Image.open(img_file)
        img = transform(img).unsqueeze(0)
        input_imgs.append(img)
    print(time.time()-stime)
    '''
    input_imgs = torch.cat(input_imgs,0)
    for i in range(18):
        stime = time.time()
        img = input_imgs.to(device)
        output = model(img)
        print('input size {} cost:{:.3}s'.format(input_imgs.size(),time.time()-stime))
    '''



if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(args.base_channel)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
       transforms.Resize((224,224)), 
       transforms.ToTensor(),
       transforms.Normalize((0.4914,0.4822,0.4405),(0.2023,0.1994,0.2010))
    ])

    main(args.img_dir,model,transform,device)

