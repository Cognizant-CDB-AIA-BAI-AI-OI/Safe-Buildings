import requests

import json

import base64

url = "http://localhost:8080/safebuild"

import base64

with open("frameImage.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

payload = {"CameraId":234545}

#files = {"frame":encoded_string}

headers = {'content-type': "image/jpeg"}

response = requests.post(url, data=encoded_string, headers=headers, stream=True)

json_data = json.loads(response.text)

print(json_data)