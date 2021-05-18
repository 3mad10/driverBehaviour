#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# In[23]:


net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg','yolov4-tiny-custom_last.weights')


# In[24]:


classes = []
with open('traffic.names','r') as f:
    classes = [line.strip() for line in f.readlines()]


# In[25]:


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[32]:


cap = cv2.VideoCapture('F:/College/Senior_2_semester_2/driver/20210502_182214.mp4')

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


# In[33]:


while True:
    _, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB = True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x+10, y + 30), font, 2, color, 2)



    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 2)
    cv2.imshow("Image", frame)
    saveFrame = 'vid/frame'+str(frame_id)+'.jpg'
    cv2.imwrite(saveFrame, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
  
cap.release()
cv2.destroyAllWindows()


# In[ ]:




