#!/bin/bash

if [ -z $CROSSBOW_HOME ]; then
    echo "error: CROSSBOW_HOME is not set"
    exit 1
fi

if [ $# -ne 2 ]; then
	echo "usage: prepare-imagenet.sh [input directory] [output directory]"
	exit 1
fi

python $CROSSBOW_HOME/scripts/datasets/imagenet/parse-records.py --subset "train"      --input-dir $1 --output-dir $2
python $CROSSBOW_HOME/scripts/datasets/imagenet/parse-records.py --subset "validation" --input-dir $1 --output-dir $2

exit 0
