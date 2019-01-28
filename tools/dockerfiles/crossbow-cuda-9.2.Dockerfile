ARG UBUNTU_VERSION=16.04

# Installation based on CUDA-9.2...

FROM nvidia/cuda:9.2-base-ubuntu${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cuda-command-line-tools-9-2 \
        cuda-cublas-9-2 \
        cuda-cufft-9-2 \
        cuda-curand-9-2 \
        cuda-cusolver-9-2 \
        cuda-cusparse-9-2 \
        libcudnn7=7.1.4.18-1+cuda9.2 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda9.1 \
# apt-get install nvinfer-runtime-trt-repo-ubuntu1604-3.0.4-ga-cuda8.0 \
RUN apt-get update && \
        apt-get update \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openjdk-8-jdk \
    maven \
    libboost-all-dev \
    graphviz \
    vim

RUN git clone http://github.com/lsds/Crossbow.git crossbow
RUN git clone https://github.com/xianyi/OpenBLAS.git openblas
WORKDIR /openblas
RUN make
WORKDIR /
WORKDIR /openblas
RUN make install
WORKDIR /
ENV BLAS_HOME /opt/OpenBLAS
ENV CUDA_HOME /usr/local/cuda
# libjpeg-turbo
#
RUN git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
WORKDIR /libjpeg-turbo
# cmake
RUN apt-get update && apt-get install -y --no-install-recommends cmake
RUN apt-get update && apt-get install -y --no-install-recommends nasm
RUN cmake -G"Unix Makefiles"
RUN make
WORKDIR /
ENV JPEG_HOME /libjpeg-turbo
ENV LD_LIBRARY_PATH $JPEG_HOME/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $BLAS_HOME/lib:$LD_LIBRARY_PATH
ENV CROSSBOW_HOME /crossbow
WORKDIR /crossbow
# NCCL?
# Don't install this?
# RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libnccl-dev
# RUN ./scripts/build.sh
RUN mvn package
WORKDIR /crossbow/clib-multigpu
RUN ./genmakefile.sh
# nvidia-cuda-dev? Need include/cublas_v2.h...
# RUN apt-get update && apt-get install -y --no-install-recommends nvidia-cuda-dev
# Not used:
# ENV NCCL_VERSION 2.2.12
# ENV CUDA_VERSION 9.1.85
ENV CUDA_PKG_VERSION 9-2
#=$CUDA_VERSION-1

# libnccl2=$NCCL_VERSION-1+cuda9.1 && \
# libnccl-dev=$NCCL_VERSION-1+cuda9.1 && \

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libnccl-dev
# cudnn.h?
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn7-dev=7.1.4.18-1+cuda9.2
RUN make -j
WORKDIR /crossbow
RUN ./scripts/build.sh
# wget no found
RUN apt-get update && apt-get install -y --no-install-recommends wget
WORKDIR /crossbow
RUN echo "Crossbow home is $CROSSBOW_HOME"
RUN mkdir -p /data/crossbow/mnist/original
RUN mkdir -p /data/crossbow/mnist/b-1024
RUN ./scripts/datasets/mnist/prepare-mnist.sh
# libjpeg.so.62: cannot open shared object
RUN echo $LD_LIBRARY_PATH 
# RUN ls $JPEG_HOME/
ENV LD_LIBRARY_PATH $JPEG_HOME:$LD_LIBRARY_PATH
RUN ./scripts/benchmarks/lenet.sh
RUN echo "* hard memlock unlimited" >>/etc/security/limits.conf
RUN echo "* soft memlock unlimited" >>/etc/security/limits.conf
