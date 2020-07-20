FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

WORKDIR /home/customai/app

RUN apt-get update && apt-get install -y python3.6 python3-pip sudo libsm6 libxext6 libxrender-dev

RUN useradd -m customai

RUN chown -R customai:customai /home/customai/

COPY --chown=customai . /home/customai/app/

USER customai

RUN cd /home/customai/app/ && pip3 install -r requirements.txt

RUN pip3 install tensornets==0.4.6 tensorboard==1.14.0

ENTRYPOINT [ "python3", "src/app.py" ]
