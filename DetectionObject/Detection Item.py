#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# In[4]:


model = cv2.dnn_DetectionModel(frozen_model, config_file)


# In[5]:


classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    


# In[6]:


print(classLabels)


# In[7]:


print(len(classLabels))


# In[8]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[9]:


cap = cv2.VideoCapture(0)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    
    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.5)
    
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0,255,0), thickness=2)
                
    cv2.imshow('video', frame)
    
    if cv2.waitKey(2) == ord('q'):
        break
cap.release()
cap.destroyAllWindows()

