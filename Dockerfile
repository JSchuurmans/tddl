FROM nvidia/cuda:10.2-runtime
WORKDIR /
RUN apt update && apt install -y --no-install-recommends \
    git build-essential \
    python3-dev python3-pip python3-setuptools
RUN pip3 -q install pip --upgrade
RUN pip3 install .[dev]

