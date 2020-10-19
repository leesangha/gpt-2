FROM leesangha/gptmodel as build

FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip \
    libglib2.0-0 

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt
RUN mkdir /app/models
COPY --from=build /app/models/ /app/models

COPY . /app

EXPOSE 80
CMD python3 app.py