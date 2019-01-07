#!/bin/bash

USAGE="usage: ./run.sh [class name]"

# Note: only one of the two variables below should be set to true
NVPROF=false # true
NVPROF_OPTS="" # "--metrics achieved_occupancy,sm_activity"
NVPROF_OUT="nvprof-resnet-no-bottleneck.nvvp"
MEMCHK=false

crossbowFileExists () {
	filename=$1 
	if [ ! -f ${filename} ]; then
		echo "error: ${filename} not found"
		exit 1
	fi
}

crossbowDirExists () {
	directory=$1 
	if [ ! -d ${directory} ]; then
		echo "error: ${directory} not found"
		exit 1
	fi
}

MVN="${HOME}/.m2/repository"

LOG4J="${MVN}/org/apache/logging/log4j"

LOG4JAPI="${LOG4J}/log4j-api/2.5/log4j-api-2.5.jar"
LOG4JCORE="${LOG4J}/log4j-core/2.5/log4j-core-2.5.jar"

OPENCV="${MVN}/opencv/opencv-library/3.2.0/opencv-library-3.2.0.jar"

IBMILOG="${MVN}/ibm/ilog/cplex/12.7.1/cplex-12.7.1.jar:${MVN}/ibm/ilog/cp/12.7.1/cp-12.7.1.jar"

CROSSBOW="lib/crossbow-0.0.1-SNAPSHOT.jar"
TESTS="target/test-classes"

crossbowFileExists ${LOG4JAPI}
crossbowFileExists ${LOG4JCORE}

crossbowFileExists ${CROSSBOW}

crossbowDirExists ${TESTS}

# Set classpath
JCP="."
JCP="${JCP}:${CROSSBOW}:${LOG4JAPI}:${LOG4JCORE}:${OPENCV}:${IBMILOG}:${TESTS}"

# OPTS="-Xloggc:test-gc.out"
OPTS="-server -XX:+UseConcMarkSweepGC -XX:NewRatio=2 -XX:SurvivorRatio=16 -Xms48g -Xmx48g"

if [ $# -lt 1 ]; then
	echo $USAGE
	exit 1
fi

CLASS=$1
CLASSFILE="${TESTS}/`echo ${CLASS} | tr '.' '/'`.class"

crossbowFileExists ${CLASSFILE}

shift 1

if [ $NVPROF = true ]; then
	nvprof $NVPROF_OPTS -o $NVPROF_OUT java $OPTS -cp $JCP $CLASS $@
elif [ $MEMCHK = true ]; then
	cuda-memcheck java $OPTS -cp $JCP $CLASS $@
else
	java $OPTS -cp $JCP $CLASS $@
fi

echo "[./run.sh] Bye."

exit 0
