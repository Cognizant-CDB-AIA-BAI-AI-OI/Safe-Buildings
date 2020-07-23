# Safe-Buildings
Monitoring Social Distancing using AI


## Docker support 
#### build docker image
docker build -f Dockerfile -t safe_building:latest .

#### run docker image
docker run -it -p 5000:5000 safe_building 

## Test App
curl -X POST http://0.0.0.0:5000/safebuild?cameraID="234545" -H "Content-Type: image/jpeg" --data-binary @frameImage.png
