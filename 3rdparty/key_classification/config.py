import os 
from easydict import EasyDict 

_C = EasyDict()
cfg = _C 
#_C.WHITE_TRAIN_DIR = '/home/data/lj/Piano/KEY_PRESS/data/rechoose/white/train'
_C.Paper_WHITE_TRAIN_DIR = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/train/white'
#_C.Paper_WHITE_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/white_train.txt'
_C.Paper_WHITE_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/white_train.npy.npz'


_C.Paper_WHITE_VAL_DIR = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/val/white'
#_C.Paper_WHITE_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/white_val.txt'
_C.Paper_WHITE_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/white_val.npy.npz'


_C.Paper_BLACK_TRAIN_DIR = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/train/black'
#_C.Paper_BLACK_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/black_train.txt'
_C.Paper_BLACK_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/black_train.npy.npz'

_C.Paper_BLACK_VAL_DIR = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/val/black'
#_C.Paper_BLACK_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/black_val.txt'
_C.Paper_BLACK_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/black_val.npy.npz'


_C.Own_WHITE_TRAIN_DIR = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/train/white'
#_C.Own_WHITE_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/white_train.txt'
_C.Own_WHITE_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/white_train.npy.npz'

_C.Own_WHITE_VAL_DIR = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/val/white'
#_C.Own_WHITE_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/white_val.txt'
_C.Own_WHITE_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/white_val.npy.npz'

_C.Own_BLACK_TRAIN_DIR = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/train/black'
#_C.Own_BLACK_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/black_train.txt'
_C.Own_BLACK_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/black_train.npy.npz'

_C.Own_BLACK_VAL_DIR = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/val/black'
#_C.Own_BLACK_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/black_val.txt'
_C.Own_BLACK_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/black_val.npy.npz'

_C.SAVE_MODEL_DIR = '/home/data/lj/Piano/KEY_PRESS/checkpoint/experment'


_C.INPUT_SIZE = {
        'white':{
            'paperdata':[112,32],
            'owndata':[60,10]
        },
        'black':{
            'paperdata':[112,32],
            'owndata':[40,10]
        }
}
       
_C.ALPHA = {
        'white':{
            'paperdata':3.0,
            'owndata':3.0
        },
        'black':{
            'paperdata':2.0,
            'owndata':2.0
        }
}

