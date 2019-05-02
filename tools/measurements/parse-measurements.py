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
    
    def isValid(self, start, end):
        if start == 0 and end == 0:
            return True
        dt = datetime.strptime(self.timestamp, "%Y/%m/%d %H:%M:%S.%f").strftime('%s.%f')
        t  = int(float(dt) * 1000)
        if t > start and t < end:
            return True
        return False

def printStats(key, values):
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


def crossbow(filename):
    # Extract start and end time of experiment
    start = 0
    end = 0
    f = open(filename, "r")
    for line in f:
        if "Start scheduling tasks at" in line:
            s = line.split(" ")
            timestamp = s[0] + " " + s[1]
            dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
            start = int(float(dt) * 1000)
        elif "Flushing left-over training tasks" in line:
            s = line.split(" ")
            timestamp = s[0] + " " + s[1]
            dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
            end = int(float(dt) * 1000)
    return start, end


def tensorflow(filename):
    # Extract start and end time of experiment
    start = 0
    end = 0
    f = open(filename, "r")
    for line in f:
        # Look for lines that start with:
        # Done warm up at [...] ms
        # Finished at [...]
        if "Done warm up" in line:
            s = line.split(" ")
            start = int(s[4])
        elif line.startswith("Finished at"):
            s = line.split(" ")
            end = int(s[2])
    return start, end


def process(filename, start, end):
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
        # Check timestamp
        if not m.isValid(start, end):
            skipped += 1
            continue
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
    print("%d lines processed (%d measurements skipped)" % (lines, skipped))
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
        printStats(key, gpuutilvalues)
        print("Memory utilization stats for GPU " + key)
        printStats(key, memutilvalues)
    print("Aggregated values")
    printStats("agg_gpu", agggpuutil)
    printStats("agg_mem", aggmemutil)
    
    return

if __name__ == "__main__":
    #
    # Check and parse command-line arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, default=None)
    parser.add_argument('--results',  type=str, required=True, default=None)
    parser.add_argument('--type',     type=str, required=True, default=None)
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.filename):
        sys.stderr.write("error: file '%s' not found" % args.filename)
        sys.exit(1)
    
    if not os.path.isfile(args.results):
        sys.stderr.write("error: file '%s' not found" % args.results)
        sys.exit(1)
    
    # Get experiment duration
    start = 0
    end = 0
    if args.type == "crossbow":
        start, end = crossbow(args.results)
    else:
        start, end = tensorflow(args.results)
    
    # Process file
    process(args.filename, start, end)
    
    print("Bye.")
    sys.exit(0)
