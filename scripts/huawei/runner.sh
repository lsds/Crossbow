#!/bin/bash

CROSSBOW_HOME=/home/work/user-job-dir/Crossbow

[ ! -d "/cache/train_dir" ] && mkdir /cache/train_dir

python $CROSSBOW_HOME/scripts/huawei/download_data.py

bash $CROSSBOW_HOME/scripts/datasets/imagenet/prepare-imagenet.sh /cache/data_dir /cache/train_dir

python $CROSSBOW_HOME/scripts/huawei/upload_data.py