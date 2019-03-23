# CrossBow Docker image based on Huawei ModelArts CUDA 9
# ModelArts example: https://github.com/huawei-clouds/modelarts-example/blob/master/CustomImage
FROM swr.cn-north-1.myhuaweicloud.com/eiwizard/custom-gpu-cuda9-inner-moxing-cp36:1.1 as base

# Fix the source lists
RUN sed -i 's/cmc-cd-mirror.rnd.huawei.com/security.ubuntu.com/g' /etc/apt/sources.list

# Replace the standard ubuntu source with Aliyun sources if buidling in mainland China
RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list \
    && sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list 

# The pip source has been pre-configured to an internal source. Roll back to public sources.
RUN rm $HOME/.pip/pip.conf

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt update && apt install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cuda9.0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7 \ 
        libnccl2 \
        cuda-command-line-tools-9-0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        git \
        wget \
        openjdk-8-jdk \
        maven \
        libboost-all-dev \
        graphviz \
        cmake \
        nasm \
        && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME /usr/local/cuda

# Install tensorflow-gpu 1.12.0 in the conda environment (pip has been redirected to conda pip)
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.10.0 # Run this if in the mainland China
# RUN pip install tensorflow-gpu==1.12.0

# OpenBLAS (TODO: install using apt install)
RUN git clone https://github.com/xianyi/OpenBLAS.git openblas \
    && cd openblas \
    && make -j $(nproc) \
    && make install
ENV BLAS_HOME /opt/OpenBLAS
ENV LD_LIBRARY_PATH $BLAS_HOME/lib:$LD_LIBRARY_PATH

# libjpeg-turbo (TODO: install using apt install)
RUN git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git \
    && cd libjpeg-turbo \
    && cmake -G"Unix Makefiles" && make -j $(nproc)
ENV JPEG_HOME /libjpeg-turbo
ENV LD_LIBRARY_PATH $JPEG_HOME/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $JPEG_HOME:$LD_LIBRARY_PATH

# Crossbow
ENV CROSSBOW_HOME /crossbow
RUN git clone http://github.com/lsds/Crossbow.git crossbow \
    && cd crossbow \
    && mvn package \
    && cd clib-multigpu \
    && ./genmakefile.sh \
    && make -j $(nproc) \
    && cd ../ \
    && ./scripts/build.sh

# Prepare a MNIST dataset local
RUN mkdir -p /data/crossbow/mnist/original \ 
    && mkdir -p /data/crossbow/mnist/b-1024 \
    && ./crossbow/scripts/datasets/mnist/prepare-mnist.sh