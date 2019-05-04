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

if [ ! -f "cifar-10-binary.tar.gz" ]; then
	# Download the file once
	wget --no-check-certificate https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
fi

echo "[Untar]"
tar zxf cifar-10-binary.tar.gz

echo "[Store]"
[ ! -d $CROSSBOW_HOME/data/cifar-10 ] && mkdir -p $CROSSBOW_HOME/data/cifar-10
mv cifar-10-batches-bin $CROSSBOW_HOME/data/cifar-10/original

rm -f cifar-10-binary.tar.gz

echo "Bye."
