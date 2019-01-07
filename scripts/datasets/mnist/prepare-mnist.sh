#!/bin/bash

if [ -z $CROSSBOW_HOME ]; then
    echo "error: CROSSBOW_HOME is not set"
    exit 1
fi

$CROSSBOW_HOME/scripts/datasets/mnist/download-mnist.sh
$CROSSBOW_HOME/scripts/datasets/mnist/preprocess-mnist.sh $1

exit 0
