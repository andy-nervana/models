FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# clean up and refresh apt-get index
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    apt-get install -y git python-virtualenv python-pip python3-dev sox libsox-dev libsox-fmt-all

# verify pip is up to date
RUN pip install pip --upgrade

ENV PYTHONPATH "${PYTHONPATH}:/mnt/repo/models/research/:/mnt/repo/models/research/slim"
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib"

# Anaconda Jazz
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate gpu_env" >> ~/.bashrc

#RUN echo "export PYTHONPATH=$PYTHONPATH:/home/models/research/:/home/models/research/slim" >> ~/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/home/models/research:/home/models/research/slim" >> ~/.bashrc

#SOURCE BASHRC?

RUN conda create -y --name gpu_env python=2.7 tensorflow=1.8.0 tensorflow-gpu=1.8.0 cudnn=7.0.5 matplotlib=2.2.2 pillow=5.1.0 cython=0.28.3

# installing python requirements
# RUN pip install tensorflow==1.5.0 \
#                tensorflow-gpu==1.5.0 \
#                tensorflow-tensorboard \
#                cython \
#                pillow \
#                lxml \
#                jupyter \
#                matplotlib

# Install interactive tools
# RUN pip install ipython ipdb

# Install stable requirements
# RUN pip install tqdm requests librosa pandas

# Install larger requirements
# RUN pip install kubernetes
# RUN pip install tensorflow
# RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
# RUN pip install torchvision
# RUN pip install git+https://github.com/pytorch/audio


# Prep for training
CMD protoc /mnt/repo/models/research/object_detection/protos/*.proto --python_out=.
# CMD export PYTHONPATH=$PYTHONPATH:/mnt/repo/models/research/:/mnt/repo/models/research/slim
# CMD export PYTHONPATH=$PYTHONPATH:/home/models/research/:/home/models/research/slim

# CMD python -c "import time; time.sleep(100000)"
