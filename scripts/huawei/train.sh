#!/bin/bash

CROSSBOW_HOME=/crossbow

cd $CROSSBOW_HOME \
    && git pull \
    && mvn package \
    && cd clib-multigpu \
    && ./genmakefile.sh \
    && make -j $(nproc) \
    && cd ../ \
    && ./scripts/build.sh

python $CROSSBOW_HOME/scripts/huawei/download_data.py

mv /home/work/user-job-dir/Crossbow-scripts/imagenet-test.metadata $CROSSBOW_HOME/data/imagenet/imagenet-test.metadata
mv /home/work/user-job-dir/Crossbow-scripts/imagenet-train.metadata $CROSSBOW_HOME/data/imagenet/imagenet-train.metadata

bash /home/work/user-job-dir/Crossbow-scripts/resnet-50.sh

python $CROSSBOW_HOME/scripts/huawei/upload_data.py