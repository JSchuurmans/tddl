FROM nvidia/cuda:10.2-runtime

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
    git build-essential \
    python3-dev python3-pip python3-setuptools
RUN pip3 -q install pip --upgrade

WORKDIR /
COPY src /src
COPY setup.py /
RUN pip3 install .[dev]
