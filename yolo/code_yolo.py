# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:07:37 2019

@author: Hari
"""

from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util1 import *
import argparse
import os 
import os.path as osp
from darknet1 import Darknet
import pickle as pkl
import pandas as pd
import random
import datetime
import csv
from pathlib import Path

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.1)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "yolov3-objv5.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3-objv5_last.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "yolo_BL.mp4", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()


#num_classes = 80
num_classes = 1
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()



def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
	
    if label in dict_label:
        dict_label[label] += 1
    else :
        dict_label[label] = 1
    return img

def avg_dicttimetarget():
    jumlah = 0
    global dict_time_target
	#rata-rata jumlah value dari dict_time_target
    for i in range(len(dict_time_target)):
        jumlah += list(dict_time_target.values())[i]
    rata = jumlah / len(dict_time_target)
	#masukan ke avg_dict_time_target
    key_waktu = time_jalan.strftime("%X")
    avg_dict_time_target[key_waktu] = int(np.floor(rata))
    # dict_time_target di deklarasi baru sebagai dictionary kosong
    dict_time_target = {}
    time_mulai = time_jalan
    return avg_dict_time_target


#Detection phase

videofile = args.videofile #or path to the video file. 

#cap = cv2.VideoCapture(videofile)  

cap = cv2.VideoCapture(0)  #for webcam

#cap = cv2.VideoCapture("rtsp://admin:admin@167.205.40.211:554/stream1")

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()
#dictionary sebelum disimpan kecsv
dict_time_target = {}
#dictionary untuk disimpan di csv
avg_dict_time_target = {}
#deklarasi waktu
time_mulai = datetime.datetime.now()




while cap.isOpened():
    ret, frame = cap.read()
    time_jalan = datetime.datetime.now()
    z = 0
    dict_label = {}
    #resize output frame opencv
    widthresize = int(frame.shape[1] * 8/10)
    heightresize = int(frame.shape[0] * 8/10)
    if widthresize <= 500 and heightresize <=230: 
        dimresize= (widthresize, heightresize)
    else: dimresize = (500, 230)

    resize = cv2.resize(frame, dimresize, interpolation= cv2.INTER_AREA)

    if ret:   
        if (len(avg_dict_time_target) == 1):
            with open(r'C:\Users\User\deep-learning-v2-pytorch\web\Flaskta\website\static\data_yolo.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time", "Target"])
                for key, value in avg_dict_time_target.items():
                    writer.writerow([key, value])	

            try : 
                dataframe = pd.read_csv(r'.\data_history_yolo.csv', index_col = False)
            except FileNotFoundError:
                with open(r'.\data_history_yolo.csv', 'a', newline='') as csvhistory:
                    writer = csv.writer(csvhistory)
                    writer.writerow(["Time", "Target"])
                   

            with open(r'.\data_history_yolo.csv', 'a', newline='') as csvhistory:
                writer = csv.writer(csvhistory)
                for key, value in avg_dict_time_target.items():
                        writer.writerow([key, value])
                
            dataframe = pd.read_csv(r'.\data_history_yolo.csv', index_col = False)
            now =  datetime.datetime.now().strftime("%H:%M:%S")
            print(datetime.datetime.strptime(now, '%H:%M:%S') - datetime.datetime.strptime(dataframe.iloc[0][0], '%H:%M:%S'))
            if datetime.datetime.strptime(now, '%H:%M:%S') - datetime.datetime.strptime(dataframe.iloc[0][0], '%H:%M:%S') > datetime.timedelta(hours=24):
                with open(r'.\data_history_yolo.csv', 'w', newline='') as csvhistory:
                    writer = csv.writer(csvhistory)
                    writer.writerow(["Time", "Target"])
                    for key, value in avg_dict_time_target.items():
                        writer.writerow([key, value])
            avg_dict_time_target.pop(list(avg_dict_time_target.keys())[0])
        

        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
        
        

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)


		# Zero detection
        if type(output) == int:
            frames += 1
			#deklarasi waktu jalan
            time_jalan = datetime.datetime.now()
			#dict "dict_time_target"
            dict_time_target[time_jalan] = 0
			
			#pembuatan dictionary "avg_dict_time_target" setiap detik
            if (time_jalan - time_mulai > datetime.timedelta(0,1)):
                avg_dict_time_target = avg_dicttimetarget()
				
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            #cv2.imshow("frame", frame)
			
			#menampilkan opencv dengan resize width dan height
            cv2.imshow("resize", resize)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue        
        

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
        
        

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        #memberikan bounding box pada frame
        list(map(lambda x: write(x, frame), output))

        

        resize = cv2.resize(frame, dimresize, interpolation= cv2.INTER_AREA)
        #print(dict_label["person"])
        
        #cv2.imshow("frame", frame)
		
		#menampilkan opencv dengan resize width dan height
        cv2.imshow("resize", resize)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        
        #print(len(output))
        
        #print(time_jalan)
		
		
        #pembuatan dictionary "time_target"
        dict_time_target[time_jalan] = dict_label["person"]
		#pembuatan dictionary "avg_dict_time_target" setiap detik
        if (time_jalan - time_mulai > datetime.timedelta(0,1)):
            avg_dict_time_target = avg_dicttimetarget()

        #print(output)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        	
        break
        