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

crossbowCreateDeviceList () {
    n=$(($1 - 1))
    # Always include device 0
    s="0"
    for i in `seq 1 $n`; do
        s="${s},${i}"
    done
    echo $s
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

CLASS="uk.ac.imperial.lsds.crossbow.ResNetv1ForCifar"
CLASSFILE="${TESTS}/`echo ${CLASS} | tr '.' '/'`.class"

resultdir="results/"
[ ! -d ${resultdir} ] && mkdir -p ${resultdir}

crossbowDirExists ${resultdir}

layers=32
dataset="cifar-10"

numgpus=2
devices="0"

momentum="0.9"
learningrate="0.1"
learningratepolicy="multistep"
learningratesteps="80,120"
learningratestepunit="epochs"
learningrategamma="0.1"
decay="0.0001"

batchsize=512

epochs=140

# Choose a synchronisation strategy
#
# updatemodel="DEFAULT"
# updatemodel="WORKER"
# updatemodel="EAMSGD"
# updatemodel="SYNCHRONOUSEAMSGD"
# updatemodel="DOWNPOUR"
# updatemodel="HOGWILD"
# updatemodel="POLYAK-RUPPERT"
updatemodel="SMA"

alpha="0.1"

numreplicas=1
wpcscale=1

# End of configuration

datadirectory=`printf "$CROSSBOW_HOME/data/cifar-10/b-%03d" $batchsize`
    
devices=`crossbowCreateDeviceList ${numgpus}`
    
echo "[INFO] Running on $numgpus devices (${devices})"
echo "[INFO] $numreplicas replicas per GPU"
    
wpc=$(($numreplicas * $numgpus * $wpcscale))
echo "[INFO] Synchronise every $wpc tasks"
    
echo "[INFO] Batch size is $batchsize"

# echo "[INFO] Train for $epochs epochs (schedule is $learningratesteps)"
echo "[INFO] Train for $epochs epochs"
	
resultfile="resnet-32.out"
	
datadirectory=`printf "$CROSSBOW_HOME/data/cifar-10/b-%03d" $batchsize`

java $OPTS -cp $JCP $CLASS \
    --data-directory ${datadirectory} \
    --display-interval 1000 \
    --display-interval-unit "tasks" \
    --cpu false \
    --gpu true \
    --number-of-task-handlers 8 \
    --number-of-callback-handlers 8 \
    --number-of-file-handlers 1 \
    --gpu-devices ${devices} \
    --wpc ${wpc} \
    --number-of-gpu-models ${numreplicas} \
    --number-of-gpu-streams ${numreplicas} \
    --training-unit "epochs" --N ${epochs} \
    --test-interval-unit "epochs" --test-interval 2 \
    --queue-measurements true \
    --tee-measurements true \
    --batch-size ${batchsize} \
    --learning-rate ${learningrate} \
    --learning-rate-decay-policy ${learningratepolicy} \
    --step-values ${learningratesteps} \
    --learning-rate-step-unit ${learningratestepunit} \
    --gamma ${learningrategamma} \
    --momentum ${momentum} \
    --weight-decay ${decay} \
    --update-model ${updatemodel} \
    --alpha ${alpha} \
    --dataset-name ${dataset} \
    --layers ${layers} \
    --task-queue-size 64 \
    --direct-scheduling true
    # Do not redirect output to a file by default
    # &> ${resultdir}/${resultfile}

echo "Done"

exit 0
