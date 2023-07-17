import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from ensemble_boxes import *

def xywh2x1y1x2y2(bbox):
    x1 = bbox[0] - bbox[2]/2
    x2 = bbox[0] + bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    y2 = bbox[1] + bbox[3]/2
    return ([x1,y1,x2,y2])

def x1y1x2y22xywh(bbox):
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return ([x,y,w,h])

IMG_PATH = r'G:\detection\VisDrone\VisDrone2019-DET-test-challenge\images'
TXT_PATH = r'E:\ICCV23'
"""

TXT_PATH格式如下：
--ICCV23
----第一个模型的推理结果
------labels
----第一个模型的推理结果
------labels
......

"""
OUT_PATH = r'E:\wbf'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


MODEL_NAME = os.listdir(TXT_PATH)
# MODEL_NAME = ['test1','test2']

# ===============================
# Default WBF config (you can change these)
iou_thr = 0.6 #0.67
skip_box_thr = 0.001
# skip_box_thr = 0.0001
# sigma = 0.1
# boxes_list, scores_list, labels_list, weights=weights,
# ===============================

image_ids = os.listdir(IMG_PATH)
for image_id in tqdm(image_ids, total=len(image_ids)):
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    for name in MODEL_NAME:
        box_list = []
        score_list = []
        label_list = []
        # houzhui = os.path.join(name,'labels')
        houzhui = os.path.join(name,image_id.replace('jpg', 'txt'))
        txt_file = os.path.join(TXT_PATH,houzhui)
        if os.path.exists(txt_file):
            txt_df = pd.read_csv(txt_file,header=None,sep=' ').values
            for row in txt_df:
                box_list.append(xywh2x1y1x2y2(row[1:5]))
                score_list.append(row[5])
                label_list.append(int(row[0]))
            boxes_list.append(box_list)
            scores_list.append(score_list)
            labels_list.append(label_list)
            weights.append(1.0)
        else:
            continue

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    out_file = open(os.path.join(OUT_PATH,image_id.replace('jpg', 'txt')), 'w')  

    for i,row in enumerate(boxes):
        img = Image.open(os.path.join(IMG_PATH, image_id))
        img_size = img.size
        w,h = img_size
        bbox = row
        x1,y1,x2,y2 = bbox
        x1 *= w 
        y1 *= h 
        x2 *= w 
        y2 *= h 
        w_ = x2-x1
        y_ = y2-y1
        out_file.write(str(int(x1))+","+str(int(y1))+","+str(int(w_))+","+str(int(y_)) + "," + str(round(scores[i],6)) + ","+str(int(labels[i]+1))+ ",-1,-1"+'\n')
    out_file.close()