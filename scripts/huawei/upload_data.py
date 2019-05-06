#!/usr/bin/env python

from __future__ import print_function

import moxing as mox
import os

if __name__ == '__main__':
    train_dir = '/cache/train_dir'
    train_url = os.environ['DLS_TRAIN_URL']
    print('INFO: Copy trained model to ' + train_url)
    mox.file.copy_parallel(train_dir, train_url)