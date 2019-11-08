import os 
import sys 
from easydict import EasyDict 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

_C = EasyDict()
cfg = _C 
# detect hand anchor config 
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detect hand config 
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05
_C.NUM_CLASSES = 5 
_C.VIS_THRESH = 0.7

# pretained model config
_C.HAND_MODEL = os.path.join(PROJECT_ROOT,'weights','det_hand.pth')
_C.KEY_WHITE_MODEL = os.path.join(PROJECT_ROOT,'weights','white_key.pth')
_C.KEY_BLACK_MODEL = os.path.join(PROJECT_ROOT,'weights','black_key.pth')

_C.MEAN = [0.45734706, 0.43338275, 0.40058118]
_C.STD = [0.23965294, 0.23532275, 0.2398498]

## key_config 
_C.KEY_NUM_CLASSES = 2 
_C.WHITE_KEY_THRESH = 0.5
_C.BLACK_KEY_THRESH = 0.5
_C.NEAR_KEY_THRESH = 0.8 

##keyboard config 
_C.KEYBOARD_MODEL = os.path.join(PROJECT_ROOT,'weights','keyboard.pth')
_C.KEYBOARD_PALETTE = [0,0,0,64,0,128]
_C.KEYBOARD_NUM_CLASSES = 2

## save file config 
_C.SAVE_IMG_DIR = os.path.join(PROJECT_ROOT,'saved')

## hand segment 
_C.HAND_SEG_MODEL = os.path.join(PROJECT_ROOT,'weights','seg_hand.pth')
_C.HAND_SEG_PALETTE = [0,0,0,64,0,128]
_C.HAND_SEG_NUM_CLASSES = 2

