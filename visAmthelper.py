import os 
import sys
import cv2 
import numpy as np 
from config import PROJECT_ROOT,cfg
sys.path.append(os.path.join(PROJECT_ROOT,'piano_utils'))
from models.model_helper import ModelProduct
from keyboard import KeyBoard
from bwlabel import BwLabel
from PIL import Image 
from tools.helper import * 
from IPython import embed 
import time 

class VisAmtHelper(object):
    def __init__(self,file_mark):
        self.keyboard = KeyBoard()
        self.modelproduct = ModelProduct()
        self.bwlabel = BwLabel()
        self.diff_img_dir,self.detect_img_dir,self.base_img_dir,self.press_img_dir = new_save_dir(cfg.SAVE_IMG_DIR,file_mark)
        self.detect_txt = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'pitch.txt')
        self.detect_video_file = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'output.mp4')

    def process_img_dir(self,img_dir):
        img_lists = [os.path.join(img_dir,x) for x in os.listdir(img_dir) 
                if x.endswith('jpg') or x.endswith('png')]
        img_lists.sort()
        base_img,keyboard_rect,start_frame = self.find_base_img(img_lists)
        if base_img is None:
            return             
        white_loc,black_boxes,total_top,total_bottom = self.find_key_loc(base_img)
        save_img = vis_white_loc_boxes(base_img,white_loc,black_boxes)
        cv2.imwrite(os.path.join(self.base_img_dir,'base.jpg'),save_img)
        count_frame = 0 
        avgtimer = timer()
        handtimer = timer()
        keypresstimer = timer()
        fps = 25.0 
        fout = open(self.detect_txt,'w')
        for img_file in img_lists:
            avgtimer.tic()
            img = Image.open(img_file)
            count_frame+=1 
            if count_frame==start_frame:
                continue 
            file_seq = os.path.basename(img_file).split('.')[0]
            opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            handtimer.tic()
            hand_boxes = self.modelproduct.detect_hand(img,keyboard_rect)
            handetime = handtimer.toc()
            if len(hand_boxes)==0:
                continue
            cur_keyboard_img = opencv_img[keyboard_rect[1]:keyboard_rect[3],keyboard_rect[0]:keyboard_rect[2]]
            keypresstimer.tic()
            keys_list,diff_img = self.keypress(base_img,
                                                cur_keyboard_img,
                                                white_loc,
                                                hand_boxes,
                                                file_seq) 
            keyetime = keypresstimer.toc()
            print('process {} total cost:{:.3}s || hand detect:{:.3}s || keypress:{:.3}s'.format(os.path.basename(img_file),
                                                                                        avgtimer.toc(),handetime,keyetime))
            save_img = vis_detect_white_key(opencv_img,
                                            hand_boxes,
                                            keys_list,
                                            white_loc,
                                            keyboard_rect,
                                            total_top,
                                            total_bottom)
            cv2.imwrite(os.path.join(self.detect_img_dir,os.path.basename(img_file)),save_img)
            cv2.imwrite(os.path.join(self.diff_img_dir,os.path.basename(img_file)),diff_img)
            save_detect_result(keys_list,fout,fps,count_frame)
        print('avg process time:{:.3}s'.format(avgtimer.elapsed()))
        img2video(self.detect_img_dir,self.detect_video_file)

    def process_video(self,video_file):
        capture = cv2.VideoCapture(video_file)
        if not capture.isOpened():
            raise ValueError('read video wrong')
        fps = capture.get(cv2.CAP_PROP_FPS)
        base_img,keyboard_rect,start_frame = self.find_video_base_img(video_file)
        if base_img is None:
            return             
        white_loc,black_boxes,total_top,total_bottom = self.find_key_loc(base_img)
        save_img = vis_white_loc_boxes(base_img,white_loc,black_boxes)
        cv2.imwrite(os.path.join(self.base_img_dir,'base.jpg'),save_img)
        count_frame = 0
        avgtimer = timer()
        handtimer = timer()
        keypresstimer = timer()
        fout = open(self.detect_txt,'w')
        while True:
            avgtimer.tic()
            ret,opencv_img = capture.read()
            count_frame+=1 
            if count_frame==start_frame:
                continue 
            if not ret:
                break 
            img = Image.fromarray(cv2.cvtColor(opencv_img,cv2.COLOR_BGR2RGB))
            handtimer.tic()
            hand_boxes = self.modelproduct.detect_hand(img,keyboard_rect)
            handetime = handtimer.toc()
            if len(hand_boxes)==0:
                continue 
            file_seq = str(count_frame).zfill(5)
            cur_keyboard_img = opencv_img[keyboard_rect[1]:keyboard_rect[3],keyboard_rect[0]:keyboard_rect[2]]
            keypresstimer.timer()
            keys_list,diff_img = self.keypress(base_img,
                                               cur_keyboard_img,
                                               white_loc,
                                               hand_boxes,
                                               file_seq)
            keyetime = keypresstimer.toc()
            print('process {}.jpg total cost:{:.3}s || hand detect:{:.3}s || keypress:{:.3}s'.format(str(count_frame).zfill(5),
                                                                                            avgtimer.toc(),handetime,keyetime))
            save_img = vis_detect_white_key(opencv_img,
                                            hand_boxes,
                                            keys_list,
                                            white_loc,
                                            keyboard_rect,
                                            total_top,
                                            total_bottom)
            cv2.imwrite(os.path.join(self.detect_img_dir,'{}.jpg'.format(str(count_frame).zfill(5))),save_img)
            cv2.imwrite(os.path.join(self.diff_img_dir,'{}.jpg'.format(str(count_frame).zfill(5))),diff_img)
            save_detect_result(keys_list,fout,fps,count_frame)
        print('avg process time:{:.3}s'.format(avgtimer.elapsed()))
        capture.release() 
        img2video(self.detect_img_dir,self.detect_video_file)

    def keypress(self,
                base_img,
                cur_img,
                white_loc,
                hand_boxes,
                file_seq):
        index_list = near_white(white_loc,hand_boxes)
        cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2GRAY)
        base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
        diff_img = cv2.absdiff(base_img,cur_img)
        diff_img = cv2.cvtColor(diff_img,cv2.COLOR_GRAY2BGR)

        h,w= diff_img.shape[:2]
        offset = 3
        whole_list = []
        for hand_list in  index_list:
            for index in range(hand_list[0]-1,hand_list[1]):
                whole_list.append(index)
        whole_list = list(set(whole_list))
        whole_list.sort()
        detect_keys_list = []
        if len(whole_list)==0:
            return detect_keys_list
        for index in whole_list:
            crop_img = diff_img[0:h,int(white_loc[index]-offset):int(white_loc[index+1]+offset)]
            input_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
            press = self.modelproduct.detect_keys(input_img)
            if press==1:
                detect_keys_list.append(index+1)
            press_path = os.path.join(self.press_img_dir,'{}_{}.jpg'.format(file_seq,index+1))
            cv2.imwrite(press_path,crop_img)
        return detect_keys_list,diff_img 

    
    def find_base_img(self,img_list):
        base_img,rect = None,None 
        count_frame = 0 
        for img_path in img_list:
            img = Image.open(img_path)
            count_frame+=1 
            rect,has_keyboard = self.keyboard.detect_keyboard(img)
            if not has_keyboard:
                continue 
            hand_boxes = self.modelproduct.detect_hand(img,rect)
            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            base_img = img[rect[1]:rect[3],rect[0]:rect[2]]
            if len(hand_boxes)==0:
                return base_img,rect,count_frame 
            if len(hand_boxes)==1:
                hand_xmin,hand_ymin = hand_boxes[0][0][0],hand_boxes[0][0][1]
                if hand_ymin>rect[3]+5:
                    return base_img,rect,count_frame 
            if len(hand_boxes)>1:
                hand_xmin1,hand_ymin1 = hand_boxes[0][0][0],hand_boxes[0][0][1]
                hand_xmin2,hand_ymin2 = hand_boxes[1][0][0],hand_boxes[1][0][1]
                if hand_ymin1>rect[3]+5 or hand_ymin2>rect[3]+5:
                    return base_img,rect,count_frame 
        return base_img, rect,count_frame 
    
    def find_key_loc(self,base_img):
        img = base_img.copy()
        white_loc,black_boxes,total_top,total_bottom = self.bwlabel.key_loc(img)
        return white_loc,black_boxes,total_top,total_bottom 

    def find_video_base_img(self,video_file):
        base_img,rect = None,None 
        count_frame = 0
        capture = cv2.VideoCapture(video_file)
        if not capture.isOpened():
            return base_img,rect,count_frame 
        while True:
            ret,frame = capture.read()
            count_frame+=1 
            if not ret:
                break 
            img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            rect,has_keyboard = self.keyboard.detect_keyboard(img)
            if not has_keyboard:
                continue
            hand_boxes = self.modelproduct.detect_hand(img,rect)
            base_img = frame[rect[1]:rect[3],rect[0]:rect[2]]
            if len(hand_boxes)==0:
                capture.release()
                return base_img,rect,count_frame 
            if len(hand_boxes)==1:
                hand_xmin,hand_ymin = hand_boxes[0][0][0],hand_boxes[0][0][1]
                if hand_ymin>rect[3]+5:
                    capture.release()
                    return base_img,rect,count_frame 
            if len(hand_boxes)>1:
                hand_xmin1,hand_ymin1 = hand_boxes[0][0][0],hand_boxes[0][0][1]
                hand_xmin2,hand_ymin2 = hand_boxes[1][0][0],hand_boxes[1][0][1]
                if hand_ymin1>rect[3]+5 or hand_ymin2>rect[3]+5:
                    capture.release()
                    return base_img,rect,count_frame 
        capture.release()
        return base_img,rect,count_frame 
