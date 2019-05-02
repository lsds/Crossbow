#!/usr/bin/python

import sys

import os
import os.path

import argparse

import numpy as np

from datetime import datetime

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
    print("%s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (key, \
          np.mean(values), \
          np.std(values), \
          min(values), \
          max(values), \
          np.percentile(values,  5), \
          np.percentile(values, 25), \
          np.percentile(values, 50), \
          np.percentile(values, 75), \
          np.percentile(values, 99)))

def process(filename, rfile):
    # Check the result file to find the point where measurements become useful
    start_task_time = 0
    r = open(rfile, "r")
    for line in r:
        if "Start scheduling tasks at" in line:
            values = line.split(" ")
            time = values[0] + " " + values[1]
            dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
            time_ms = int(float(dt) * 1000)
            start_task_time = time_ms
            break
    print("Start time is %d" % start_task_time)
    
    f = open(filename, "r")
    lines = 0
    measurements = {}
    skipped = 0
    for line in f:
        lines += 1
        # Skip header for now
        if line.startswith("timestamp") or line.startswith("#"):
            continue
        s = line.split(",")
        time = s[0].strip()
        dt = datetime.strptime(time, "%Y/%m/%d %H:%M:%S.%f").strftime('%s.%f')
        time_ms = int(float(dt) * 1000)
        
        if time_ms < start_task_time:
            skipped += 1
            continue
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
    print("%d lines processed (%d skipped)" % (lines, skipped))
    keys = measurements.keys()
    # print(keys)
    print("%d GPU devices" % len(keys))
    K = []
    for key in keys:
        length = len(measurements[key])
        # print("%s: %d measurements" % (key, length))
        K.append(length)
    if len(set(K)) != 1:
        sys.stderr.write("error: different number of measurements per GPU")
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
        agggpuutil.extend(gpuutilvalues)
        aggmemutil.extend(memutilvalues)
        print("GPU utilization stats for GPU " + key)
        print_stats(key, gpuutilvalues)
        print("Memory utilization stats for GPU " + key)
        print_stats(key, memutilvalues)
    print("Aggregated values")
    print_stats("agg_gpu", agggpuutil)
    print_stats("agg_mem", aggmemutil)
    
    return

if __name__ == "__main__":
    #
    # Check and parse command-line arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, default=None)
    
    args = parser.parse_args()
    
    # Check if file exists
    # Assume result file has the same name as measurements file
    rfile = "./results/" + args.filename.split(".")[0] + ".out"
    if not os.path.isfile(args.filename) or not os.path.isfile(rfile):
        sys.stderr.write("error: file '%s' not found" % args.filename)
        sys.exit(1)
    
    # Process file
    process(args.filename, rfile)
    
    print("Bye.")
    sys.exit(0)
