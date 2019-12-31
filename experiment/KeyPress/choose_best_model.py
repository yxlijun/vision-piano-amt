import os 
import cv2
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from PIL import Image 
import argparse 
import torch.nn.functional as F 
from models.resnet18 import ResNet18
from models.simple import SimpleNet 
from config import cfg 

parser = argparse.ArgumentParser()
parser.add_argument('--type',default='white',choices=['white','black'])
parser.add_argument('--img_dir','-id',type=str,default=None)
parser.add_argument('--model_dir','-md',type=str)
parser.add_argument('--pos',type=str,default='True')
parser.add_argument('--network',type=str,default='simple',choices=['simple','resnet'])
args = parser.parse_args()

def str2bool(param):
    return param in ['True','1','true','yes']

num_classes = 2
base_channel = 16
if args.network=='simple':
    if args.type=='black':
        input_h,input_w = cfg.BLACK_INPUT
    else:
        input_h,input_w = cfg.WHITE_INPUT 
else:
    input_h,input_w = 224,224
transform = transforms.Compose([
        transforms.Grayscale(),
	transforms.Resize((input_h, input_w)),
	transforms.ToTensor()
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def list_img_dir(path,img_lists):
    subpaths = [os.path.join(path,x) for x in os.listdir(path)]
    for subpath in subpaths:
        if os.path.isdir(subpath):
            list_img_dir(subpath,img_lists)
        elif subpath.endswith('jpg'):
            img_lists.append(subpath)


def list_models(path):
	model_files = [os.path.join(path,x) for x in os.listdir(path) if 'epoch' in x ]
	model_files.sort()
	return model_files


def detect_white_keys(model,imgs):
    inputs = list()
    with torch.no_grad():
        for img in imgs:
            img = transform(img).unsqueeze(0)
            img = img.cuda()
            inputs.append(img)
        inputs = torch.cat(inputs,dim=0)
        output = model(inputs)
        prob = F.softmax(output, dim=1)   #----按行softmax,行的和概率为1,每个元素代表着概
        prob = prob.cpu().numpy()
        result = (prob[:,1]>0.5).astype(int)
    return result,prob[:,1]
    
def detect_black_keys(model,imgs):
    inputs = list()
    with torch.no_grad():
        for img in imgs:
            img = transform(img).unsqueeze(0)
            img = img.cuda()
            inputs.append(img)
        inputs = torch.cat(inputs,dim=0)
        output = model(inputs)
        prob = F.softmax(output,dim=1)
        prob = prob.cpu().numpy()
        result = (prob[:,1]>0.5).astype(int)
    return result,prob[:,1] 

def main():
    img_lists = []
    list_img_dir(args.img_dir,img_lists)
    model_files = list_models(args.model_dir)
    for model_file in model_files:
        model = ResNet18(base_channel,num_classes)
        if args.network=='simple':
            model = SimpleNet(type=args.type)
        else:
            model = ResNet18(base_channel,num_classes)
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)
        model.eval()
        count = 0
        for img_path in img_lists:
            input_imgs = []
            img = Image.open(img_path)
            input_imgs.append(img)
            if args.type=='black':
                pred,prob = detect_black_keys(model,input_imgs)
            else:
                pred,prob = detect_white_keys(model,input_imgs)
            if str2bool(args.pos):
                if pred[0]==1:count+=1
            else:
                if pred[0]==0:count+=1
        print('{} {}/{}={}'.format(model_file,count,len(img_lists),count/len(img_lists)))


if __name__=='__main__':
	main()
