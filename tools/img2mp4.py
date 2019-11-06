import os
import cv2
from PIL import Image
import numpy as np

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
size = (1280,720)
vw = cv2.VideoWriter('output.mp4', fourcc=fourcc, fps=25.0, frameSize=size)
path = '/home/data/cy/projects/piano/data/res/press_imgs/level_7_no_2'
img_files = [os.path.join(path,x) for x in os.listdir(path)]
img_files.sort()
for file in img_files:
    f_read = cv2.imread(file)
    f_img = Image.fromarray(f_read)  
    f_rs = f_img.resize([1280,720],resample=Image.NONE)
    f_out = np.array(f_rs)
    vw.write(f_out)
vw.release()