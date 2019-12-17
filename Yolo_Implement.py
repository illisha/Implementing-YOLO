#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import numpy as np


# In[3]:


net = cv2.dnn.readNet("C://Users//PC//Downloads//yolov3.weights", "C://Users//PC//Downloads//yolov3.cfg")


# In[4]:


classes = []


# In[5]:


with open("C://Users//PC//Downloads//coco.names","r") as f:
     classes = [line.strip() for line in f.readlines()]


# In[6]:


print(classes)


# In[7]:


#get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[16]:


img = cv2.imread("C://Users//PC//Downloads//DSC_0054.jpg")
img = cv2.resize(img, None, fx=0.18, fy=0.18)
height, width, channels = img.shape


# In[17]:


blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)


# In[18]:


class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
print(len(indexes))

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




