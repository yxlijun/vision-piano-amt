import os
import cv2
from PIL import Image
import numpy as np
import argparse 

def img2video(path,size=(1280,720)):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw = cv2.VideoWriter('output.mp4', fourcc=fourcc, fps=25.0, frameSize=size)
    img_files = [os.path.join(path,x) for x in os.listdir(path)]
    img_files.sort()
    for file in img_files:
        f_read = cv2.imread(file)
        f_img = Image.fromarray(f_read)  
        f_rs = f_img.resize(list(size),resample=Image.NONE)
        f_out = np.array(f_rs)
        vw.write(f_out)
    vw.release()



def readvideo2flipvideo(video_path,save_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError('read video wrong')
    fps = capture.get(cv2.CAP_PROP_FPS)

    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    save_mp4 = os.path.join(save_path,os.path.basename(video_path))
    videowriter = cv2.VideoWriter(save_mp4,fourcc=fourcc,fps=fps,frameSize=size)
    while True:
        ret,frame = capture.read()
        if not ret:
            break 
        frame = cv2.flip(frame,-1)
        f_img = Image.fromarray(frame)
        f_re = f_img.resize(list(size),resample=Image.NONE)
        f_out = np.array(f_re)
        videowriter.write(f_out)

    videowriter.release()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video','-v',type=str,help='video')
    parser.add_argument('--save_path','-s',type=str,default='/home/lj/project/piano_keys/visAmt/test_imgs/paperData')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    readvideo2flipvideo(args.video,args.save_path)
