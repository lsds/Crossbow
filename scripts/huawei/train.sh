#!/bin/bash

CROSSBOW_HOME=/crossbow

python $CROSSBOW_HOME/scripts/huawei/download_data.py

# mv /home/work/user-job-dir/Crossbow-scripts/imagenet-test.metadata $CROSSBOW_HOME/data/imagenet/imagenet-test.metadata
# mv /home/work/user-job-dir/Crossbow-scripts/imagenet-train.metadata $CROSSBOW_HOME/data/imagenet/imagenet-train.metadata

mv /home/work/user-job-dir/Crossbow-scripts/imagenet-test.metadata /cache/data_dir/
mv /home/work/user-job-dir/Crossbow-scripts/imagenet-train.metadata /cache/data_dir/

bash /home/work/user-job-dir/Crossbow-scripts/resnet-50.sh

python $CROSSBOW_HOME/scripts/huawei/upload_data.py