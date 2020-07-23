from werkzeug.wrappers import Request, Response
from flask import Flask
import cv2
import os
import io
import time
import json
import numpy as np
from flask import Flask, url_for
from flask import request
from flask import Response
from flask import jsonify
import base64
import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time


app = Flask(__name__)

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
sess = tf.Session()
sess.run(model.pretrained())

def points_on_Layout_view(pedestrian_boxes, M, Outdata):
    # function to map points on Floor Plan Layout
    for i in range(len(pedestrian_boxes)):
        mid_point_x = int(
            (pedestrian_boxes[i][0] + pedestrian_boxes[i][2] ) / 2
        )

        mid_point_y = pedestrian_boxes[i][3]

        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]

        Outdata['PeopleDetails'].append({'X':int(warped_pt[0]),'Y':int(warped_pt[1])})
    return Outdata


def get_prediction(frame):
    # TO-DO build model once
    # Yolo V3 model defination
    

    classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
    list_of_classes=[0,1,2,3,5,7]

    img = frame
    imge=np.array(img).reshape(-1,416,416,3)

    # function to calculate no. person detected and their bounding box
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
    
    # sess.close()
    return pedestrian_boxes,total_pedestrians


# Process frame, and map coordinates
def personDetect(frame, cameraid, configdata):
    frame = cv2.resize(frame, (416,416))

    # load input camera and layout details in json format
    for j,k in enumerate(configdata['Camera']):
        if str(k["CameraId"]) == cameraid:
            data= configdata['Camera'][j]
        else:
            break

    srcCordinate = []
    dstCordinate = []

    #camera four coordinates
    for i in data['Areas']['areas'][0]['CameraCoordinates']:
        srcCordinate.append((int(i['X']),int(i['Y'])))

    #FloorPlanCoordinates
    for i in data['Areas']['areas'][0]['FloorPlanCoordinates']:
        dstCordinate.append((int(i['X']),int(i['Y'])))
    
    Outdata = {"CameraId": 0,"TimeStamp":"2019-07-11T03:54:16.000Z"}

    Outdata['CameraId'] = data['CameraId']

    Outdata['PeopleDetails'] = []
    Outdata['PeopleCount'] = 0


    # source and destination point declaration
    src = np.float32(np.array(srcCordinate))
    dst = np.float32(np.array(dstCordinate))

    # determination of transformation matrix
    M,a = cv2.findHomography(src, dst)
    pedestrian_detect = frame


    # Detect person and bounding boxes using DNN
    start_time=time.time()
    pedestrian_boxes, num_pedestrians = get_prediction(frame)
    print("--- %s seconds ---" % (time.time() - start_time))

    Outdata['PeopleCount']= len(pedestrian_boxes)

    # map point cordinates, when there is person
    if len(pedestrian_boxes) > 0:
        Outdata = points_on_Layout_view(pedestrian_boxes, M,Outdata)
        return Outdata


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/safebuild', methods=['POST'])
def safebuild():
    try:
        if (request.args):
            try:
                cameraID = request.args.get('cameraID')              
            except Exception as ex:
                print('EXCEPTION:', str(ex))   

        imageData = io.BytesIO(request.get_data())
        processed_img = cv2.imdecode(np.frombuffer(imageData.getbuffer(), np.uint8), -1)

        # load input camera and layout details in json format
        with open('app/config.json') as f:
            configdata = json.load(f)

        outputJson = personDetect(processed_img, cameraID, configdata)

        print("outputJson", outputJson)
        
        return jsonify(outputJson)  

    except Exception as e:
        print('EXCEPTION:', str(e))
        return Response(response='Error processing image ' + str(e), status= 500)


if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=False)