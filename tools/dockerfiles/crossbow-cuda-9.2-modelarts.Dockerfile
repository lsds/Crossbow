# CrossBow Docker image based on Huawei ModelArts CUDA 9.2
# ModelArts example: https://github.com/huawei-clouds/modelarts-example/blob/master/CustomImage
FROM swr.cn-north-1.myhuaweicloud.com/eiwizard/custom-gpu-cuda92-base:1.0 as base

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
        libcudnn7 \ 
        libnccl2 \ 
        libnccl-dev \
        libcudnn7-dev \
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

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Prepare a MNIST dataset local
RUN mkdir -p /data/crossbow/mnist/original \ 
    && mkdir -p /data/crossbow/mnist/b-1024 \
    && ./crossbow/scripts/datasets/mnist/prepare-mnist.sh