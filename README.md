# Safe-Buildings
Monitoring Social Distancing using AI


## docker support 
#### build docker image
docker build -f Dockerfile -t safe_building:latest .

#### run docker image
docker run -it safe_building 

#### run docker image and connect to app 
docker run -it -p 5000:5000 safe_building

# command to Run the code

python3.6 app/main_v2.py --videopath https://votttest.blob.core.windows.net/vottstorage/GroceryStore2.mp4 --coordinateJson inputcordinate.json --outputJson output.json --outputCycle 30
