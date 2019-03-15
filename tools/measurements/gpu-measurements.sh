#!/bin/bash
#
# GPU query metrics
# 
# timestamp:          Timestamp, in "YYYY/MM/DD HH:MM:SS.msec" format.
# gpu_serial:         The serial number of each GPU, a globally unique 
#                     immutable alphanumeric identifier.
# utilization.gpu:    Percentage of time over the past sampling period 
#                     during which one or more kernels were executing. 
# utilization.memory: Percentage of time over the past sampling period 
#                     during which the global (device) memory was read
#                     or written.
#
# Other metrics include:
#
# name
# gpu_name
# gpu_uuid
# pstate
# temperature_gpu
# memory.total
# memory.free
# memory.used
# pcie.link.gen.max
# pcie.link.gen.current
# gpu_bus_id
# pci.bus_id
# vbios_version
# driver_version
#
# Assemble metrics
#
METRICS=
METRICS="$METRICS,timestamp"
METRICS="$METRICS,gpu_serial"
METRICS="$METRICS,temperature.gpu"
METRICS="$METRICS,utilization.gpu"
METRICS="$METRICS,utilization.memory"
METRICS="$METRICS,memory.used"
METRICS="$METRICS,memory.free"

# Set duration (between 1/6 and 1 seconds)
UNIT="ms"
DURATION="200" # 1/5 of a second

# Set format
FORMAT="csv,nounits"

# Try filename
FILENAME="measurements.csv"
if [ -f $FILENAME ]; then
    echo "error: $FILENAME already exists"
    exit 1
fi

# Run the command
echo "Writing to $FILENAME..."
echo "# `date`" >$FILENAME
nvidia-smi --query-gpu=${METRICS} -l${UNIT} ${DURATION} --format=${FORMAT} >>${FILENAME} 2>&1

echo "Bye."
exit 0
