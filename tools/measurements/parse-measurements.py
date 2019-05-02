#!/usr/bin/python

import sys

import os
import os.path

import argparse

import numpy as np

class GPU(object):
    
    def __init__(self):
        self.temperature = 0
        self.utilisation = 0

class MemoryInfo(object):
    
    def __init__(self):
        self.utilisation = 0
        self.used = 0
        self.free = 0
    
class Measurement(object):
    
    def __init__(self):
        self.key = None
        self.timestamp = None
        self.gpu = GPU()
        self.memory = MemoryInfo()

def print_stats(key, values):
    print(key, \
          np.mean(values), \
          np.std(values), \
          min(values), \
          max(values), \
          np.percentile(values,  5), \
          np.percentile(values, 25), \
          np.percentile(values, 50), \
          np.percentile(values, 75), \
          np.percentile(values, 99))

def process(filename):
    f = open(filename, "r")
    lines = 0
    measurements = {}
    for line in f:
        lines += 1
        # Skip header for now
        if line.startswith("timestamp") or line.startswith("#"):
            continue
        s = line.split(",")
        m = Measurement()
        # 
        # 0: timestamp          (date/time)
        # 1: serial             (string)
        # 2: GPU temperature    (C)
        # 3: GPU utilization    (%), 
        # 4: memory utilization (%), 
        # 5: memory used        (MB), 
        # 6: memory free        (MB)
        #
        m.timestamp = s[0].strip()
        m.serial = s[1].strip()
        # GPU
        m.gpu.temperature = float(s[2])
        m.gpu.utilisation = float(s[3])
        # Memory
        m.memory.utilisation = float(s[4])
        m.memory.used = float(s[5])
        m.memory.free = float(s[6])
        # Append to dictionary
        measurements.setdefault(m.serial, []).append(m)
    print("%d lines processed" % lines)
    keys = measurements.keys()
    # print(keys)
    print("%d GPU devices" % len(keys))
    K = []
    for key in keys:
        length = len(measurements[key])
        # print("%s: %d measurements" % (key, length))
        K.append(length)
    if len(set(K)) != 1:
        print("error: different number of measurements per GPU", file=sys.stderr)
        sys.exit(1)
    print("%d measurements per GPU" % K[0])
    
    # Perform some statistics
    agggpuutil = []
    aggmemutil = []
    for key in keys:
        if len(measurements[key]) == 0:
            continue
        gpuutilvalues = []
        memutilvalues = []
        for m in measurements[key]:
            gpuutilvalues.append(m.gpu.utilisation)
            memutilvalues.append(m.memory.utilisation)
        if np.mean(gpuutilvalues) < 1:
            continue
        else:
            agggpuutil.extend(gpuutilvalues)
            aggmemutil.extend(memutilvalues)
        print_stats(key, gpuutilvalues)
        print_stats(key, memutilvalues)
    print("Aggregated values")
  
    
    
    return

if __name__ == "__main__":
    #
    # Check and parse command-line arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, default=None)
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.filename):
        print("error: file '%s' not found" % filename, file=sys.stderr)
        sys.exit(1)
    
    # Process file
    process(args.filename)
    
    print("Bye.")
    sys.exit(0)
