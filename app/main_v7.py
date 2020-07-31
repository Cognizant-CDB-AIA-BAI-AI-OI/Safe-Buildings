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
from datetime import datetime


app = Flask(__name__)

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
sess = tf.Session()
sess.run(model.pretrained())

def points_on_Layout_view(pedestrian_boxes, M, Outdata):
    # function to map points on Floor Plan Layout
    for i in pedestrian_boxes:
        pts = np.array([[[i[0], i[1]]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]

        Outdata['peopledetails'].append({'x':int(warped_pt[0]),'y':int(warped_pt[1])})
    return Outdata

# A utility function to calculate area of triangle
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0) 

# A function to check whether point P(x, y) lies inside the rectangle 
def check(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
    A1 = area(x, y, x1, y1, x2, y2)
    A2 = area(x, y, x2, y2, x3, y3)
    A3 = area(x, y, x3, y3, x4, y4)
    A4 = area(x, y, x1, y1, x4, y4)
    print("A1 + A2 + A3 + A4,A",A,A1 + A2 + A3 + A4)
    if (abs((A1 + A2 + A3 + A4)-A)<3000):
        return 1
    else:
        return 0

def findTransformationMatrix(data):
    srcCordinate = [0]*4
    dstCordinate = [0]*4

    #camera four coordinates
    for i in data['cameraCoordinates']:
        # for maintaining the sequence of coordinate
        if (str(i['vertex']) =="A"):
            srcCordinate[0]= (int(i['x']),int(i['y']))
        if (str(i['vertex']) =="B"):
            srcCordinate[1]= (int(i['x']),int(i['y']))
        if (str(i['vertex']) =="C"):
            srcCordinate[2]= (int(i['x']),int(i['y']))
        if (str(i['vertex']) =="D"):
            srcCordinate[3]= (int(i['x']),int(i['y']))


    #FloorPlanCoordinates
    for i in data['floorPlanCoordinates']:
        # for maintaining the sequence of coordinate
        if (str(i['vertex']) =="A"):
            dstCordinate[0]= (int(i['x']),int(i['y']))
        if (str(i['vertex']) =="B"):
            dstCordinate[1]= (int(i['x']),int(i['y']))
        if (str(i['vertex']) =="C"):
            dstCordinate[2]= (int(i['x']),int(i['y']))
        if (str(i['vertex']) =="D"):
            dstCordinate[3]= (int(i['x']),int(i['y']))


    # source and destination point declaration
    src = np.float32(np.array(srcCordinate))
    dst = np.float32(np.array(dstCordinate))

    # determination of transformation matrix
    M,a = cv2.findHomography(src, dst)

    return M



def get_prediction(frame):
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
    for k in configdata['cameraRules']:
        if (str(k["cameraId"]) == cameraid):
            data= k

    # Detect person and bounding boxes using DNN
    start_time=time.time()
    pedestrian_boxes, num_pedestrians = get_prediction(frame)
    

    camera_areas = []
    for j in data['areas']['areas']:
        temp_area_pts = []
        for i in range(len(pedestrian_boxes)):
            x = int((pedestrian_boxes[i][0] + pedestrian_boxes[i][2] ) / 2)
            y = int(pedestrian_boxes[i][3])
            print("(x,y)",x,y)
            x1,x2,x3,x4 = [int(i['x']) for i in j['cameraCoordinates']]
            y1,y2,y3,y4 = [int(i['y']) for i in j['cameraCoordinates']]

            print("values",x1, y1, x2, y2, x3, y3, x4, y4, x, y)
            if (check(x1, y1, x2, y2, x3, y3, x4, y4, x, y)):
                print("yes")
                temp_area_pts.append((x,y))
        
        camera_areas.append(temp_area_pts)
    
    print("camera_areas",camera_areas)
        
    tstmp = datetime.utcnow().isoformat()[:-3] + 'Z'
    Output = {"Output":[]}
    Outdata = {"cameraId": 0,"timestamp":str(tstmp)}

    Outdata['cameraId'] = cameraid

    Outdata['peopledetails'] = []
    Outdata['peoplecount'] = 0

    for m,n  in enumerate(camera_areas):
        M = findTransformationMatrix(data['areas']['areas'][m])
        Outdata['areaid'] = data['areas']['areas'][m]['areaId']
        # output json initiation
        Outdata['peoplecount']= len(n)
        print("Outdata['peoplecount']",Outdata['peoplecount'])
        # map point cordinates, when there is person
        if len(n) > 0:
            Outdata = points_on_Layout_view(n, M,Outdata)
            print(Outdata)
            Output['Output'].append(Outdata)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return Output


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
        with open('config_v5.json') as f:
            configdata = json.load(f)

        outputJson = personDetect(processed_img, cameraID, configdata)

        print("outputJson", outputJson)
        
        return jsonify(outputJson)  

    except Exception as e:
        print('EXCEPTION:', str(e))
        return Response(response='Error processing image ' + str(e), status= 500)


if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=False)