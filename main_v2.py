import cv2
import os
import argparse
import YoloModel
import time
import json
import urllib.request
import numpy as np


# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Command-line input setup
parser = argparse.ArgumentParser(description="SocialDistancing")
parser.add_argument(
    "--videopath", type=str, default="GroceryStore2.mp4", help="Path to the video file"
)
parser.add_argument(
    "--coordinateJson", type=str, default="inputcordinate.json", help="Path to the video file"
)

parser.add_argument(
    "--outputJson", type=str, default="output.json", help="Path to the video file"
)

parser.add_argument(
    "--outputCycle", type=str, default="30", help="time to get output"
)

args = parser.parse_args()

input_video_url = args.videopath
input_video = input_video_url.split('/')[-1]
urllib.request.urlretrieve(input_video_url, input_video)

inputJsonpath = args.coordinateJson
outputJsonpath = args.outputJson

outputCycle = args.outputCycle

# load input camera and layout details in json format
with open(inputJsonpath) as f:
  data = json.load(f)

srcCordinate = []
dstCordinate = []

#camera four coordinates
for i in data['Areas']['areas'][0]['CameraCoordinates']:
    srcCordinate.append((int(i['X']),int(i['Y'])))

#FloorPlanCoordinates
for i in data['Areas']['areas'][0]['FloorPlanCoordinates']:
    dstCordinate.append((int(i['X']),int(i['Y'])))

print(srcCordinate,dstCordinate)

with open(outputJsonpath) as f:
  Outdata = json.load(f)
  Outdata['CameraId'] = data['CameraId']
  Outdata['SpaceId'] = data['SpaceId']
  Outdata['AreaId'] = data['Areas']['areas'][0]['AreaLabel']

# Get video handle
cap = cv2.VideoCapture(input_video)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frameCount = fps*int(outputCycle)

print("fps,frameCount", fps,frameCount)

# Initialize necessary variables
frame_num = 0

# Process each frame, until end of video
while cap.isOpened():
    frame_num += 1
    ret, frame = cap.read()

    if not ret:
        print("end of the video file...")
        break

    frame = cv2.resize(frame,(416,416))
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    if ((frame_num%frameCount ==0) or (frame_num == 1)):
        Outdata['PeopleDetails'] = []
        Outdata['PeopleCount'] = 0

        if frame_num == 1:

            # source and destination point declaration
            src = np.float32(np.array(srcCordinate))
            dst = np.float32(np.array(dstCordinate))

            # determination of transformation matrix
            M,a = cv2.findHomography(src, dst)
            pedestrian_detect = frame


        # Detect person and bounding boxes using DNN
        start_time=time.time()
        pedestrian_boxes, num_pedestrians = YoloModel.pedDetection(frame)
        print("--- %s seconds ---" % (time.time() - start_time))
        Outdata['PeopleCount']= len(pedestrian_boxes)


        # function to map points on Floor Plan Layout
        def points_on_Layout_view(pedestrian_boxes, M):

            for i in range(len(pedestrian_boxes)):

                mid_point_x = int(
                    (pedestrian_boxes[i][0] + pedestrian_boxes[i][2] ) / 2
                )

                mid_point_y = pedestrian_boxes[i][3]

                pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
                warped_pt = cv2.perspectiveTransform(pts, M)[0][0]

                Outdata['PeopleDetails'].append({'X':int(warped_pt[0]),'Y':int(warped_pt[1])})

        # map point cordinates, when there is person
        if len(pedestrian_boxes) > 0:
            points_on_Layout_view(pedestrian_boxes, M)
            print("output Json", Outdata)




