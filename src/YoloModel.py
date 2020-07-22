import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

# Yolo V3 model defination
inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)

classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
list_of_classes=[0,1,2,3,5,7]
sess = tf.Session()

# function to calculate no. person detected and their bounding box
def pedDetection(frame):
    sess.run(model.pretrained())
    img = frame
    imge=np.array(img).reshape(-1,416,416,3)
    preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
    boxes = model.get_boxes(preds, imge.shape[1:3])
    boxes1=np.array(boxes)
    pedestrian_boxes = []
    total_pedestrians = 0
    for j in list_of_classes:
        count =0
        if str(j) in classes:
            lab=classes[str(j)]
        if (len(boxes1) !=0 and lab == 'person'):   
            for i in range(len(boxes1[j])):
                box=boxes1[j][i] 
                if boxes1[j][i][4]>=.10:        
                    count += 1
                    total_pedestrians = total_pedestrians + 1
                    pedestrian_boxes.append(box)

    return pedestrian_boxes,total_pedestrians





