import os 
import sys
import argparse 
from visAmthelper import VisAmtHelper 

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',type=str,default=None)
parser.add_argument('--video',type=str,default=None)
args = parser.parse_args()


if __name__=='__main__': 
    if args.img_dir is not None:
        if args.img_dir[-1]=='/':
            args.img_dir = args.img_dir[0:-1]
        file_mark = args.img_dir.split('/')[-1]
        visamthelper = VisAmtHelper(file_mark)
        visamthelper.process_img_dir(args.img_dir) 
    if args.video is not None:
        file_mark = os.path.basename(args.video).split('.')[0]
        visamthelper = VisAmtHelper(file_mark)
        visamthelper.process_video(args.video)
