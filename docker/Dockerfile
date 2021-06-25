# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Build this Dockerfile by running the following commands:
#
#     $ cd /path/to/your/trt_pose
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     $ docker/set_nvidia_runtime.sh
#

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

WORKDIR trt_pose

ARG JUPYTER_PASSWORD=jetson
ENV JUPYTER_PASSWORD=${JUPYTER_PASSWORD}

#
# install pre-requisite packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            cmake \
            python3-opencv \
            curl \
    && rm -rf /var/lib/apt/lists/*

# pip dependencies for pytorch-ssd
RUN pip3 install --verbose --upgrade Cython && \
    pip3 install --verbose boto3 pandas

# pip dependencies for trt_pose
RUN pip3 install tqdm cython pycocotools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*


# =======================================
# INSTALL OPENCV (with GStreamer support)
# =======================================
RUN echo -e "\e[48;5;20m Install OpenCV \e[0m"
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            libopencv-python \
    && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && rm -rf /var/lib/apt/lists/*


# =================
# INSTALL GSTREAMER
# =================
RUN apt-get update && \
    apt-get install -y \
    libwayland-egl1 \
    gstreamer1.0-plugins-bad \
    libgstreamer-plugins-bad1.0-0 \
    gstreamer1.0-plugins-good \
    python3-gst-1.0


# =======================
# INSTALL JUPYTER RELATED
# =======================

# Install traitlets (master, to support the unlink() method)
RUN echo -e "\e[48;5;172m Install traitlets \e[0m"
#sudo python3 -m pip install git+https://github.com/ipython/traitlets@master
RUN python3 -m pip install git+https://github.com/ipython/traitlets@dead2b8cdde5913572254cf6dc70b5a6065b86f8

# Install jupyter lab
RUN echo -e "\e[48;5;172m Install Jupyter Lab \e[0m"
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs libffi-dev 
RUN pip3 install jupyter jupyterlab==2.2.6 --verbose && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter lab --generate-config && \
    python3 -c "from notebook.auth.security import set_password; set_password('${JUPYTER_PASSWORD}', '/root/.jupyter/jupyter_notebook_config.json')"    

# Install jupyter_clickable_image_widget
RUN echo "\e[42m Install jupyter_clickable_image_widget \e[0m"
RUN cd && \
    apt-get install -y libssl1.0-dev && \
    git clone https://github.com/jaybdub/jupyter_clickable_image_widget && \
    cd jupyter_clickable_image_widget && \
    git checkout tags/v0.1 && \
    pip3 install -e . && \
    jupyter labextension install js && \
    jupyter lab build

# =================
# INSTALL torch2trt
# =================
ENV TORCH2TRT_REPO_DIR=/opt/
RUN cd ${TORCH2TRT_REPO_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install

# ==============
# INSTALL jetcam
# ==============
ENV JETCAM_REPO_DIR=/opt/
RUN cd ${JETCAM_REPO_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/jetcam && \
    cd jetcam && \
    python3 setup.py install

# ================
# INSTALL trt_pose
# ================
ENV TRTPOSE_REPO_DIR=/opt/trt_pose
COPY . ${TRTPOSE_REPO_DIR}
RUN cd ${TRTPOSE_REPO_DIR} && python3 setup.py install

# ================
# Pre-cache models 
# ================
RUN python3 -c "import torchvision; \
                model = torchvision.models.resnet18(pretrained=True); \
                model = torchvision.models.resnet34(pretrained=True); \
                model = torchvision.models.resnet50(pretrained=True); \
                model = torchvision.models.resnet101(pretrained=True); \
                model = torchvision.models.resnet152(pretrained=True)"

#
# install other packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            vim \
    && rm -rf /var/lib/apt/lists/*

