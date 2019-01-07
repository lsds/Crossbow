#!/usr/bin/env sh
#
# This scripts downloads the mnist data and unzips it
# (based on a similar script from Caffe)

if [ -z $CROSSBOW_HOME ]; then
    echo "error: CROSSBOW_HOME is not set"
    exit 1
fi

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "[Download]"

wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "[Unzip]"

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "[Store]"
[ ! -d $CROSSBOW_HOME/data/mnist/original ] && mkdir -p $CROSSBOW_HOME/data/mnist/original
mv *-images-*-ubyte $CROSSBOW_HOME/data/mnist/original
mv *-labels-*-ubyte $CROSSBOW_HOME/data/mnist/original

echo "Bye."
