# Crossbow Docker image based on nvidia docker CUDA 9.2
ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:9.2-base-ubuntu${UBUNTU_VERSION} as base

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt update && apt install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cuda-command-line-tools-9-2 \
        cuda-cublas-9-2 \
        cuda-cufft-9-2 \
        cuda-curand-9-2 \
        cuda-cusolver-9-2 \
        cuda-cusparse-9-2 \
        libcudnn7=7.1.4.18-1+cuda9.2 \ 
        libnccl2 \ 
        libnccl-dev \
        libcudnn7-dev=7.1.4.18-1+cuda9.2 \
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
        cuda-libraries-dev-9-2 \
        cuda-nvml-dev-9-2 \
        cuda-minimal-build-9-2 \
        cuda-command-line-tools-9-2 \
        python3 \
        python3-pip  \
        && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME /usr/local/cuda

# OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git openblas \
    && cd openblas \
    && git checkout v0.3.5 \
    && make -j $(nproc) \
    && make install \
    && cd ../ \
    && rm -fr openblas
ENV BLAS_HOME /opt/OpenBLAS
ENV LD_LIBRARY_PATH $BLAS_HOME/lib:$LD_LIBRARY_PATH

# libjpeg-turbo
RUN git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git \
    && cd libjpeg-turbo \
    && git checkout 2.0.1 \
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

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python \
    && pip3 install setuptools \
    && pip3 install tensorflow==1.12.0

# Prepare a MNIST dataset local
RUN mkdir -p /data/crossbow/mnist/original \ 
    && mkdir -p /data/crossbow/mnist/b-1024 \
    && ./crossbow/scripts/datasets/mnist/prepare-mnist.sh