FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# FROM nvidia/cuda:9.0-runtime-ubuntu16.04
# ARG repository
# FROM nvidia/cuda:9.0-devel-ubuntu16.04
# LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
# 
# ENV CUDNN_VERSION 7.0.5.15
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
# 
# RUN apt-get update && apt-get install -y --no-install-recommends \
#             libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
#             libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
#     rm -rf /var/lib/apt/lists/*

# FROM tensorflow/tensorflow:1.8.0-gpu

# RUN apt-get install cuda-libraries-9-0

# FROM nvidia/cuda:9.0-base-ubuntu16.04

# WORKDIR /
# COPY cudnn-9.0-linux-x64-v6.0.tgz /
# RUN tar -xzvf /cudnn-8.0-linux-x64-v6.0.tgz && \
#     mkdir -p /usr/local/cuda/include && \
#     mkdir -p /usr/local/cuda/lib64 && \
#     cp cuda/include/cudnn.h /usr/local/cuda/include && \
#     cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
#     chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# FROM ubuntu:16.04

# WORKDIR /
# COPY cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb /

# RUN dpkg -i /cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
# RUN apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
# RUN apt-get update
# RUN apt-get -y --allow-unauthenticated install cuda-9.0

# FROM itamarost/object-detection-app:latest-gpu

# clean up and refresh apt-get index
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# make sure pip is at latest version
RUN apt-get update && \
    apt-get install -y python-pip && \
    apt-get install -y python3-pip && \
    apt-get clean autoclean && \
    apt-get autoremove -y
RUN pip install pip --upgrade && pip3 install pip --upgrade

# install python requirements
# ADD requirements.txt /tmp/requirements.txt
# RUN pip3 install -r /tmp/requirements.txt
RUN pip install tensorflow-gpu==1.5.0 tensorflow-tensorboard Cython pillow lxml jupyter matplotlib

# RUN apt-get install python-support
# RUN update-python-modules -a
# RUN apt-get -y install python3-tk 

# add source code
ADD . /root/src
WORKDIR /root/src

ENV PYTHONPATH "${PYTHONPATH}:/mnt/repo/models/research/:/mnt/repo/models/research/slim"
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib"

# run something
# CMD protoc /mnt/repo/models/research/object_detection/protos/*.proto --python_out=.
# CMD export PYTHONPATH=$PYTHONPATH:/mnt/repo/models/research/:/mnt/repo/models/research/slim
# CMD python3 /mnt/repo/models/research/object_detection/train.py --logtostderr --pipeline_config_path=$CONFIG_PATH --train_dir=$TRAIN_PATH
CMD python -c "import time; time.sleep(1000)"
