import cv2
import os
import time

import json
import numpy as np
from flask import Flask, url_for
from flask import request
from flask import Response
from flask import jsonify
import base64

app = Flask(__name__)


# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# function to map points on Floor Plan Layout
def points_on_Layout_view(pedestrian_boxes, M,Outdata):
    

    for i in range(len(pedestrian_boxes)):

        mid_point_x = int(
            (pedestrian_boxes[i][0] + pedestrian_boxes[i][2] ) / 2
        )

        mid_point_y = pedestrian_boxes[i][3]

        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]

        Outdata['PeopleDetails'].append({'X':int(warped_pt[0]),'Y':int(warped_pt[1])})
        
    return Outdata


# Process frame, and map coordinates
def personDetect(frame,cameraid):
    frame = cv2.resize(frame,(416,416))
    
    # load input camera and layout details in json format
    with open('config.json') as f:
      configdata = json.load(f)

    for j,k in enumerate(configdata['Camera']):
        if k["CameraId"] == cameraid:
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

    print(srcCordinate,dstCordinate)
    
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
    import YoloModel
    pedestrian_boxes, num_pedestrians = YoloModel.pedDetection(frame)
    print("--- %s seconds ---" % (time.time() - start_time))
    Outdata['PeopleCount']= len(pedestrian_boxes)

    # map point cordinates, when there is person
    if len(pedestrian_boxes) > 0:
        Outdata = points_on_Layout_view(pedestrian_boxes, M,Outdata)
        print("output Json", Outdata)
        return Outdata

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/safebuild', methods = ['POST'])
def safebuild():
    if request.headers['Content-Type'] == 'image/jpeg':
        cameraid  = 234545
        print("cam id",cameraid)
        image = request.data# raw data with base64 encoding
        
        decoded_data = base64.b64decode(image)
        np_data = np.fromstring(decoded_data,np.uint8)
        img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
        outputJson = personDetect(img,cameraid)
        print("outputJson",outputJson)

        return jsonify(outputJson)

#host='0.0.0.0',
if __name__ == '__main__':
    app.run(debug=True,port=8080)

