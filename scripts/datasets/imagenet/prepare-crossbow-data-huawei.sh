#!/bin/bash

CROSSBOW_HOME=/home/work/user-job-dir/Crossbow
cd $CROSSBOW_HOME/scripts/datasets/imagenet/

[ ! -d "/cache/train_dir" ] && mkdir /cache/train_dir

python download_data.py

bash prepare-imagenet.sh /cache/data_dir /cache/train_dir

python upload_data.py