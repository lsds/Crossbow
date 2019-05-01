#!/bin/bash

if [ -z $CROSSBOW_HOME ]; then
    echo "error: CROSSBOW_HOME is not set"
    exit 1
fi

$CROSSBOW_HOME/scripts/datasets/mnist/download-cifar-10.sh
$CROSSBOW_HOME/scripts/datasets/mnist/preprocess-cifar-10.sh $1

exit 0
