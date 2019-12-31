import os
import cv2
from PIL import Image
import numpy as np
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',type=str,default=None)
args = parser.parse_args()


def img2video(img_dir,video_file):
    imgs = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if x.endswith('jpg')]
    assert len(imgs)>0,print('imgs error')
    img = cv2.imread(imgs[0])
    h,w,_ = img.shape 
    size = (w,h)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(video_file,fourcc=fourcc, fps=25.0, frameSize=size)
    imgs.sort()
    for img_file in tqdm(imgs):
        img = cv2.imread(img_file)
        fimg = Image.fromarray(img)  
        fimg = fimg.resize(list(size),resample=Image.NONE)
        img = np.array(fimg)
        videoWriter.write(img)
    videoWriter.release()


        

if __name__=='__main__':
    if args.img_dir is not None:
        img_dir = args.img_dir 
        if img_dir[-1]=='/':img_dir = img_dir[0:-1]
        video_file = os.path.join(os.path.split(img_dir)[0],'output.mp4')
        img2video(img_dir,video_file)
