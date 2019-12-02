#-*- coding:utf-8 -*-
import cv2
import numpy
import argparse
import os
import shutil
import mido
from IPython import embed 

class Accuracy(object):
    def __init__(self, 
                midiPath, 
                pframe_time=0.05,
                start_frame = 0,
                midi_offset = 0):
        self.midiPath = midiPath
        self.pframe_time = pframe_time 
        self.offTime = start_frame * pframe_time
        self.pitch_onset = self.processMidi(self.midiPath,self.offTime,midi_offset)  

        self.black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                          31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                          60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
        self.white_num = [x for x in range(1, 89) if x not in self.black_num]
        

    def processMidi(self, midiPath, offTime, midi_offset):
        mid = mido.MidiFile(midiPath)
        timeCount = 0
        dataList = []
        for msg in mid:
            if not msg.is_meta:
                if msg.type == 'control_change':
                    timeCount = timeCount + msg.time
                elif msg.type == 'note_on' or msg.type == 'note_off':
                    timeCount = timeCount + msg.time
                    data = [msg.type, msg.note - 20, msg.velocity, timeCount]
                    dataList.append(data)
        dict1 = {}
        result = []
        for data in dataList:
            if data[0] == 'note_on' and data[2] > 0:
                dict1[data[1]] = data[1:]
            else:
                dict1[data[1]].append(data[3])
                result.append(dict1.pop(data[1]))
        result = sorted(result, key = lambda x : x[2])
        pitch_onset = []
        for item in result:
            po = [item[2] - midi_offset + offTime, item[0]]
            pitch_onset.append(po)
        pitch_onset = sorted(pitch_onset, key=lambda x: (x[0], x[1]))
        return pitch_onset
    
        
if __name__ == '__main__':
    EVALUATE_MAP = { 
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
    Acu = Accuracy(midiPath, w_detectPath, b_detectPath,start_frame=173,pframe_time=1/24.0)
