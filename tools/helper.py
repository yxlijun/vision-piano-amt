import cv2
import numpy as np 
import os 
import time 
from PIL import Image 

def vis_bw_key(img,
        white_loc,
        black_boxes,
        white_top_boxes,
        white_bottom_boxes):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    
    for box in black_boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)

    for box in white_top_boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(255,0,255),1)

    for box in white_bottom_boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(255,255,0),2)

    return img_copy

def vis_white_loc(img,white_loc):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    return img_copy 

def vis_boxes(img,boxes,color): 
    img_copy = img.copy()
    for box in boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),color,2)
    return img_copy 

def vis_white_loc_boxes(img,white_loc,boxes):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    for box in boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
    return img_copy 

def vis_detect_white_key(img,
                        hand_boxes,
                        key_indexs,
                        white_loc,
                        rect,
                        total_top,
                        total_bottom):
    img_copy = img.copy()        
    for ind,loc in enumerate(white_loc):
        if ind ==len(white_loc)-1:
            break 
        cv2.putText(img_copy, str(ind + 1), (int(white_loc[ind] + 2+rect[0]), int(0.9 *(rect[3]-rect[1]) +rect[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    for index in key_indexs:
        box1 = total_top[index-1]
        box2 = total_bottom[index-1]
        x1_loc = int(box1[0]+rect[0])
        y1_loc = int(box1[1]+rect[1])
        x2_loc = int(box2[0]+rect[0])
        y2_loc = int(box2[1]+rect[1])
        cv2.rectangle(img_copy, (x1_loc, y1_loc), (x1_loc + int(box1[2]), y1_loc + int(box1[3])), (0, 0, 255), 1)
        cv2.rectangle(img_copy, (x2_loc, y2_loc), (x2_loc + int(box2[2]), y2_loc + int(box2[3])), (0, 0, 255), 1)
    for box in hand_boxes:
        cv2.rectangle(img_copy,box[0],box[1],(0,255,0),2)
    return img_copy 


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def new_save_dir(root,file_mark):
    diff_img_dir = os.path.join(root,file_mark,'diff_img')
    detect_img_dir = os.path.join(root,file_mark,'detect_img')
    base_img_dir = os.path.join(root,file_mark,'base_img')
    press_img_dir = os.path.join(root,file_mark,'press_img')
    ensure_dir(diff_img_dir)
    ensure_dir(detect_img_dir)
    ensure_dir(base_img_dir)
    ensure_dir(press_img_dir)
    return diff_img_dir,detect_img_dir,base_img_dir,press_img_dir

def near_white(white_loc,boxes):
    index_list = []
    if len(boxes)==0:
        return index_list 
    for box in boxes:
        min_loc = box[0][0]
        max_loc = box[1][0]
        diffs1,diffs2 = [],[]
        for w_loc in white_loc:
            diff1 = abs(min_loc-w_loc)
            diff2 = abs(max_loc-w_loc)
            diffs1.append(diff1)
            diffs2.append(diff2)
        left_index = diffs1.index(min(diffs1))-1 
        right_index = diffs2.index(min(diffs2))+1
        index_list.append((left_index,right_index))
    return index_list 
    
class timer(object):
    def __init__(self):
        self.count = 0 
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0 
        self.avg_time = 0 
    
    def reset():
        self.start_time = 0
        self.end_time = 0 
        self.total_time = 0
        self.avg_time = 0 
        self.count = 0

    def tic(self):
        self.start_time = time.time()
    
    def toc(self):
        self.end_time = time.time()
        self.total_time+=(self.end_time-self.start_time)
        self.count+=1 
        return self.end_time-self.start_time 

    def elapsed(self,avg=True):
        if avg:
            return self.total_time/self.count 
        return self.total_time 

def save_detect_result(keys_list,fout,fps,count_frame):
    if len(keys_list)==0:
        return 
    per_frame_time = 1.0/fps 
    cur_time = per_frame_time*(count_frame-1)
    fout.write('{:.4} '.format(cur_time))
    for key in keys_list:
        fout.write('{} '.format(key))
    fout.write('\n')
    fout.flush()


def img2video(img_dir,video_file):
    size = (1280,720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(video_file,fourcc=fourcc, fps=25.0, frameSize=size)
    imgs = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if x.endswith('jpg')]
    imgs.sort()
    for img_file in imgs:
        img = cv2.imread(img_file)
        fimg = Image.fromarray(img)  
        fimg = fimg.resize(list(size),resample=Image.NONE)
        img = np.array(fimg)
        videoWriter.write(img)
    videoWriter.release()

