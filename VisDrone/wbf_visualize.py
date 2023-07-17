import os  
from PIL import Image
import cv2 


# wbf后的结果
labels = r'G:\wbf\wbf'
# 原图路径
images = r'G:\detection\VisDrone\VisDrone2019-DET-test-challenge\images'
# 可视化保存的路径
out = r'G:\wbf\result'
# 不同label的映射，采用的YOLOv5中VisDrone的颜色
colors = [(56,56,255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (207,210,49), (10,249,72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0)]

i = 0
for label in os.listdir(labels):
    i += 1
    image = os.path.join(images,label.replace('txt','jpg'))
    out_img = os.path.join(out,label.replace('txt','jpg'))
    label = os.path.join(labels,label)
    
    img = cv2.imread(image)
    h,w,_ = img.shape
    with open(label,'r') as f:
        for line in f.readlines():
            x1,y1,w_,h_,_,cls,_,_ = line.split(',')
            cls = int(cls)
            x1 = int(x1)
            y1 = int(y1)
            w_ = int(w_)
            h_ = int(h_)            
            x2 = x1+w_
            y2 = y1+h_
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls-1], 2)
            cv2.imwrite(out_img,img)
            