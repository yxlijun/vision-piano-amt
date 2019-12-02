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
_C.KEY_WHITE_MODEL = os.path.join(PROJECT_ROOT,'weights','white_key4.pth')
_C.KEY_BLACK_MODEL = os.path.join(PROJECT_ROOT,'weights','black_key2.pth')

_C.MEAN = [0.45734706, 0.43338275, 0.40058118]
_C.STD = [0.23965294, 0.23532275, 0.2398498]

## key_config 
_C.KEY_NUM_CLASSES = 2 
_C.WHITE_KEY_THRESH = 0.6
_C.BLACK_KEY_THRESH = 0.6
_C.NEAR_KEY_THRESH = 0.9
_C.WHITE_BASE_CHANNEL = 16
_C.BLACK_BASE_CHANNEL = 16
##keyboard config 
_C.KEYBOARD_MODEL = os.path.join(PROJECT_ROOT,'weights','keyboard1.pth')
_C.KEYBOARD_PALETTE = [0,0,0,64,0,128]
_C.KEYBOARD_NUM_CLASSES = 2

## save file config 
_C.SAVE_IMG_DIR = os.path.join('/home/data/lj/Piano','saved')

## hand segment 
_C.HAND_SEG_MODEL = os.path.join(PROJECT_ROOT,'weights','seg_hand1.pth')
_C.HAND_SEG_PALETTE = [0,0,0,64,0,128]
_C.HAND_SEG_NUM_CLASSES = 2

### vision 
_C.VISION_DETECT = True 

###evalute
_C.EVALUATE_MAP = {
        'originWhite':{'start_frame':245,'midi':'/home/data/gxdata/selfRecord/1120/originWhite.MID','fps':24,'midi_offset':1.5},
        'originBlack':{'start_frame':156,'midi':'/home/data/gxdata/selfRecord/1120/originBlack.MID','fps':24,'midi_offset':1.5},
        'ExtraLightBlack':{'start_frame':97,'midi':'/home/data/gxdata/selfRecord/1120/ExtraLightBlack.MID','fps':24,'midi_offset':1.5},
        'ExtraLightWhite':{'start_frame':142,'midi':'/home/data/gxdata/selfRecord/1120/ExtraLightWhite.MID','fps':24,'midi_offset':1.5},

        'extraLightWhite2':{'start_frame':105,'midi':'/home/data/gxdata/selfRecord/1121/extraLightWhite2.MID','fps':24,'midi_offset':1.5}, 
        'extraLightBlack2':{'start_frame':83,'midi':'/home/data/gxdata/selfRecord/1121/extraLightBlack2.MID','fps':24,'midi_offset':1.5}, 
        'random1':{'start_frame':164,'midi':'/home/data/gxdata/selfRecord/1121/random1.MID','fps':24,'midi_offset':1.5}, 
        'random2':{'start_frame':174,'midi':'/home/data/gxdata/selfRecord/1121/random2.MID','fps':24,'midi_offset':1.5}, 
        'random3':{'start_frame':161,'midi':'/home/data/gxdata/selfRecord/1121/random3.MID','fps':24,'midi_offset':1.5}, 
        'random4':{'start_frame':173,'midi':'/home/data/gxdata/selfRecord/1121/random4.MID','fps':24,'midi_offset':1.5}, 
        'x_lulu_-15':{'start_frame':121,'midi':'/home/data/gxdata/selfRecord/1121/x_lulu_-15.MID','fps':24,'midi_offset':1.5}, 
        'x_mainchord_-15':{'start_frame':104,'midi':'/home/data/gxdata/selfRecord/1121/x_mainchord_-15.MID','fps':24,'midi_offset':1.5}, 
        'x_random5_+15':{'start_frame':34,'midi':'/home/data/gxdata/selfRecord/1121/x_random5_+15.MID','fps':24,'midi_offset':1.5},

        'blackWhite':{'start_frame':52,'midi':'/home/data/gxdata/selfRecord/1126/blackWhite.MID','fps':25,'midi_offset':1.5},
        'Chord':{'start_frame':93,'midi':'/home/data/gxdata/selfRecord/1126/Chord.MID','fps':25,'midi_offset':1.5},

        'V1':{'start_frame':98,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V1.wmv.mid','fps':20,'midi_offset':0}, 
        'V2':{'start_frame':72,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V2.wmv.mid','fps':20,'midi_offset':0}, 
        'V3':{'start_frame':51,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V3.wmv.mid','fps':20,'midi_offset':0}, 
        'V4':{'start_frame':75,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V4.wmv.mid','fps':20,'midi_offset':0}, 
        'V5':{'start_frame':110,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V5.wmv.mid','fps':20,'midi_offset':0}, 
        'V6':{'start_frame':101,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V6.wmv.mid','fps':20,'midi_offset':0}, 
        'V7':{'start_frame':87,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V7.wmv.mid','fps':20,'midi_offset':0}, 
        'V8':{'start_frame':94,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V8.wmv.mid','fps':20,'midi_offset':0}, 
        'V9':{'start_frame':122,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V9.wmv.mid','fps':20,'midi_offset':0}, 
        'V10':{'start_frame':85,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V10.wmv.mid','fps':20,'midi_offset':0}, 
}
