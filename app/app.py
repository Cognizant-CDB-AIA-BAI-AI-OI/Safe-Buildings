import io
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

import tensornets as nets
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
sess = tf.Session()
sess.run(model.pretrained())


def points_on_Layout_view(N, M):
    # function to map points on Floor Plan Layout
    pDetails = []
    for i in N:
        pts = np.array([[[i[0], i[1]]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        pDetails.append({'x': int(warped_pt[0]), 'y': int(warped_pt[1])})
    return pDetails


def findTransformationMatrix(data):
    srcCordinate = [0] * 4
    dstCordinate = [0] * 4

    # camera coordinates
    for i in data['cameraCoordinates']:
        # for maintaining the sequence of coordinate
        if (str(i['vertex']) == "A"):
            srcCordinate[0] = (int(i['x']), int(i['y']))
        if (str(i['vertex']) == "B"):
            srcCordinate[1] = (int(i['x']), int(i['y']))
        if (str(i['vertex']) == "C"):
            srcCordinate[2] = (int(i['x']), int(i['y']))
        if (str(i['vertex']) == "D"):
            srcCordinate[3] = (int(i['x']), int(i['y']))

    # FloorPlanCoordinates
    for i in data['floorPlanCoordinates']:
        # for maintaining the sequence of coordinate
        if (str(i['vertex']) == "A"):
            dstCordinate[0] = (int(i['x']), int(i['y']))
        if (str(i['vertex']) == "B"):
            dstCordinate[1] = (int(i['x']), int(i['y']))
        if (str(i['vertex']) == "C"):
            dstCordinate[2] = (int(i['x']), int(i['y']))
        if (str(i['vertex']) == "D"):
            dstCordinate[3] = (int(i['x']), int(i['y']))

    print("srcCordinate", srcCordinate)
    print("dstCordinate", dstCordinate)

    # source and destination point declaration
    src = np.float32(np.array(srcCordinate))
    dst = np.float32(np.array(dstCordinate))

    # determination of transformation matrix
    M, a = cv2.findHomography(src, dst)
    return M


def get_prediction(frame):
    classes = {
        '0': 'person',
        '1': 'bicycle',
        '2': 'car',
        '3': 'bike',
        '5': 'bus',
        '7': 'truck'
    }
    list_of_classes = [0, 1, 2, 3, 5, 7]

    img = frame
    imge = np.array(img).reshape(-1, 416, 416, 3)

    # function to calculate no. person detected and their bounding box
    preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
    boxes = model.get_boxes(preds, imge.shape[1:3])
    boxes1 = np.array(boxes)
    pedestrian_boxes = []
    total_pedestrians = 0

    for j in list_of_classes:
        count = 0
        if str(j) in classes:
            lab = classes[str(j)]
        if (len(boxes1) != 0 and lab == 'person'):
            for i in range(len(boxes1[j])):
                box = boxes1[j][i]
                if boxes1[j][i][4] >= .40:
                    count += 1
                    total_pedestrians = total_pedestrians + 1
                    pedestrian_boxes.append(box)

    return pedestrian_boxes, total_pedestrians


def personDetect(img, cameraid, configdata):
    # Process frame, and map coordinates
    frame = cv2.resize(img, (416, 416))
    scale_h = img.shape[0] / 416
    scale_w = img.shape[1] / 416
    PCameraID = False

    # load input camera and layout details in json format
    for k in configdata['cameraRules']:
        if (str(k["cameraId"]) == cameraid):
            data = k
            PCameraID = True
            break

    if (PCameraID):
        # Detect person and bounding boxes using DNN
        start_time = time.time()
        pedestrian_boxes, num_pedestrians = get_prediction(frame)

        print("value of num_pedestrians", num_pedestrians)
        camera_areas = []
        for j in data['areas']['areas']:
            temp_area_pts = []
            for i in range(len(pedestrian_boxes)):
                x = int(
                    int((pedestrian_boxes[i][0] + pedestrian_boxes[i][2]) / 2)
                    * scale_w)
                y = int(int(pedestrian_boxes[i][3]) * scale_h)
                print("(x,y)", x, y)
                srcCordinate_C = [0] * 4
                # camera coordinates
                for i in j['cameraCoordinates']:
                    # for maintaining the sequence of coordinate
                    if (str(i['vertex']) == "A"):
                        srcCordinate_C[0] = [int(i['x']), int(i['y'])]
                    if (str(i['vertex']) == "B"):
                        srcCordinate_C[1] = [int(i['x']), int(i['y'])]
                    if (str(i['vertex']) == "C"):
                        srcCordinate_C[2] = [int(i['x']), int(i['y'])]
                    if (str(i['vertex']) == "D"):
                        srcCordinate_C[3] = [int(i['x']), int(i['y'])]

                contPts = np.array(srcCordinate_C)
                print("values", contPts, x, y)
                inPolflag = int(cv2.pointPolygonTest(contPts, (x, y), False))
                print("flag value inPolflag", inPolflag)
                if (inPolflag == 1):
                    print("yes")
                    temp_area_pts.append((x, y))

            camera_areas.append(temp_area_pts)

        print("camera_areas", camera_areas)

        output = []

        for m, n in enumerate(camera_areas):
            Outdata = {}
            tstmp = datetime.utcnow().isoformat()[:-3] + 'Z'
            Outdata['timestamp'] = str(tstmp)
            Outdata['cameraId'] = cameraid
            M = findTransformationMatrix(data['areas']['areas'][m])
            Outdata['areaid'] = data['areas']['areas'][m]['areaId']
            # output json initiation
            Outdata['peoplecount'] = len(n)
            print("n###############", n)
            # map point cordinates, when there is person
            outdata = points_on_Layout_view(n, M)
            Outdata['peopledetails'] = outdata
            print(Outdata)
            output.append(Outdata)

        Output = {"PeopleTelemetryData": output}
        print("--- %s seconds ---" % (time.time() - start_time))
        return Output
    else:
        return {"Output": "Camera Id not found"}


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
        processed_img = cv2.imdecode(
            np.frombuffer(imageData.getbuffer(), np.uint8), -1)

        # load input camera and layout details in json format
        with open('app/config_v6.json') as f:
            configdata = json.load(f)

        outputJson = personDetect(processed_img, cameraID, configdata)
        print("outputJson", outputJson)
        return jsonify(outputJson)

    except Exception as e:
        print('EXCEPTION:', str(e))
        return Response(response='Error processing image ' + str(e),
                        status=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
