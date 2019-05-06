#!/usr/bin/env python

from __future__ import print_function

import moxing as mox
import time
import os

if __name__ == '__main__':
    data_dir = '/cache/data_dir'
    start = time.time()
    data_url = os.environ['DLS_DATA_URL']
    print('INFO: Start copying data from the blob storage ' + data_url + ' into SSD under ' + data_dir)
    mox.file.copy_parallel(data_url, data_dir)
    print('INFO: Copying completes! The copy task takes: ' + str(time.time() - start) + ' seconds')