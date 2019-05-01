#!/bin/bash

USAGE="usage: [TBD]"

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

crossbowFileExists ${LOG4JAPI}
crossbowFileExists ${LOG4JCORE}

if [ -z $CROSSBOW_HOME ]; then
    echo "error: CROSSBOW_HOME is not set"
    exit 1
fi

CROSSBOW="${CROSSBOW_HOME}/target/crossbow-0.0.1-SNAPSHOT.jar"
TESTS="${CROSSBOW_HOME}/target/test-classes"

crossbowFileExists ${CROSSBOW}
crossbowDirExists ${TESTS}

# Set classpath
JCP="."
JCP="${JCP}:${CROSSBOW}:${LOG4JAPI}:${LOG4JCORE}:${TESTS}"

# OPTS="-Xloggc:test-gc.out"
OPTS="-server -XX:+UseConcMarkSweepGC -XX:NewRatio=2 -XX:SurvivorRatio=16 -Xms48g -Xmx48g"

CLASS="uk.ac.imperial.lsds.crossbow.preprocess.cifar.Cifar"
CLASSFILE="${TESTS}/`echo ${CLASS} | tr '.' '/'`.class"

batchsize=$1
if [ -z "$batchsize" ]; then
    batchsize=1
fi

inputdirectory="$CROSSBOW_HOME/data/cifar-10/original"
outputdirectory=`printf "$CROSSBOW_HOME/data/cifar-10/b-%03d" $batchsize`

if [ -d "$outputdirectory" ]; then
    echo "error: $outputdirectory already exists"
    exit 1
fi
mkdir -p $outputdirectory

java $OPTS -cp $JCP $CLASS -i $inputdirectory -o $outputdirectory -b $batchsize

echo "Done"

exit 0
