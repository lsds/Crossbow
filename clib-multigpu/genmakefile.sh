#!/bin/sh
#
# usage: ./genmakefile.sh
#
# Shell script that generates the Crossbow C/C++ library Makefile
# for the present system
#
# If a Makefile exists, it is renamed to Makefile.save

# Check if CROSSBOW_HOME is set
if [ -z "$CROSSBOW_HOME" ]; then
	echo "error: \$CROSSBOW_HOME is not set"
	exit 1
fi

# Source common functions
. "$CROSSBOW_HOME"/scripts/common.sh

# Source configuration parameters
. "$CROSSBOW_HOME"/scripts/crossbow.conf

MAKEFILE="$CROSSBOW_HOME"/clib-multigpu/Makefile

# Find OS
OS=`uname -s`
if [ \( "$OS" != "Darwin" \) -a \( "$OS" != "Linux" \) ]; then
	echo "error: unsupported operating system: $OS"
	exit 1
fi

JH="$JAVA_HOME"
[ -z "$JH" ] && { # If JAVA_HOME is not set, try to find it
[ "$OS" = "Darwin" ] && {
	
	crossbowProgramExistsOrExit "/usr/libexec/java_home"
	JH=`/usr/libexec/java_home`
	
} || { # OS is Linux
	
	crossbowProgramExistsOrExit "readlink"
	JH=`readlink -f $(which java) | awk '{ split($0, t, "/jre/"); print t[1] }'`
}
}

# Check Java home
crossbowDirectoryExistsOrExit "$JH"
# Check if jni.h exists
# crossbowFileExistsOrExit "$JH/include/jni.h"

CH="$CUDA_HOME"
# If CUDA_HOME is not set, try to find it
[ -z "$CH" ] && [ -d "/usr/local/cuda" ] && CH="/usr/local/cuda"

BH="$BLAS_HOME"
# If BLAS_HOME is not set, try to find it
[ -z "$BH" ] && [ -d "/opt/OpenBLAS" ] && BH="/opt/OpenBLAS"

NH="$NCCL_HOME"
# If NCCL_HOME is not set, try to find it
[ -z "$NH" ] && [ -d "/opt/nccl" ] && BH="/opt/nccl"

TH="$JPEG_HOME"
# If JPEG_HOME is not set, try to find it
[ -z "$TH" ] && [ -d "/opt/libjpeg-turbo" ] && TH="/opt/libjpeg-turbo"

#
# With all the necessary variables in place,
# create Makefile
#
[ -e "$MAKEFILE" ] && mv "$MAKEFILE" "$MAKEFILE".save

touch "$MAKEFILE"

echo "# Makefile for Crossbow C/C++ library" >>"$MAKEFILE"
echo "# Customised for `uname -n` running `uname -s` on `date`" >>"$MAKEFILE"
echo "#" >>"$MAKEFILE"

# Set operating system
echo "OS = $OS" >>"$MAKEFILE"

# Set paths to libraries:
echo "CUDA_PATH := $CH" >>"$MAKEFILE"
echo "JAVA_PATH := $JH" >>"$MAKEFILE"
echo "BLAS_PATH := $BH" >>"$MAKEFILE"
echo "NCCL_PATH := $NH" >>"$MAKEFILE"
echo "JPEG_PATH := $TH" >>"$MAKEFILE"
echo ""
echo "CBOW_PATH := $CROSSBOW_HOME" >> "$MAKEFILE"
echo ""

cat <<!endoftemplate! >>"$MAKEFILE"

CLASS_PATH := ../target/classes
vpath %.class \$(CLASS_PATH)

ARCH := \$(shell uname -m)
ifneq (,\$(filter \$(ARCH),x86_64))
	TARGETSIZE := 64
else
	\$(error error: unsupported architecture \$(ARCH))
endif

OS := \$(shell uname -s)
ifeq (,\$(filter \$(OS),Linux Darwin))
	\$(error error: unsupported OS \$(OS))
endif

CC := cc
NV := \$(CUDA_PATH)/bin/nvcc -ccbin \$(CC)

# So far, used for libRNG
CPP := g++

#
# Optional
#
DBG := -g -G
# Added to supress warnings after switch to CUDA 8.0
WARN := -Wno-deprecated-gpu-targets

NVFLAGS := -m\$(TARGETSIZE) \$(DBG) \$(WARN)
CCFLAGS :=
LDFLAGS :=

ifeq (\$(OS), Darwin)
	# CCFLAGS += -rpath \$(CUDA_PATH)/lib
	LDFLAGS += -arch \$(ARCH)
endif

#
# Required for building dynamic libraries
#
CFLBASE := --compiler-options '-W -Wall -DWARNING -fPIC -Wno-unused-function -march=native -O3'

CFL := \$(CFLBASE)
CFL += \$(NVFLAGS)
CFL += \$(addprefix -Xcompiler , \$(CCFLAGS))

LFL := \$(CFL)
LFL += \$(addprefix -Xlinker , \$(LDFLAGS))

ifeq (\$(OS), Darwin)
	LFL += -Xlinker -framework -Xlinker CUDA
endif

INCLUDES := -I/usr/include -D_GNU_SOURCE

# CUDA
INCLUDES += -I\$(CUDA_PATH)/include

# OpenBLAS
ifneq (\$(BLAS_PATH),)
	INCLUDES += -I\$(BLAS_PATH)/include
endif

# NCCL
ifneq (\$(NCCL_PATH),)
	INCLUDES += -I\$(NCCL_PATH)/include
endif

# Turbo-JPEG
ifneq (\$(JPEG_PATH),)
	INCLUDES += -I\$(JPEG_PATH)
endif

# JNI
ifeq (\$(OS),Darwin)
	INCLUDES += -I/Library/Java/JavaVirtualMachines/jdk1.8.0_45.jdk/Contents/Home/include
	INCLUDES += -I/Library/Java/JavaVirtualMachines/jdk1.8.0_45.jdk/Contents/Home/include/darwin
else
	INCLUDES += -I\$(JAVA_PATH)/include
	INCLUDES += -I\$(JAVA_PATH)/include/linux
endif

LIBS += -lpthread

# Turbo-JPEG (or JPEG)

LIBS += -L\$(JPEG_PATH) -ljpeg

# CUDA
ifneq (\$(OS),Darwin)
	POSTFIX := 64
endif
LIBS += -L\$(CUDA_PATH)/lib\$(POSTFIX) -lcudart -lcublas -lcudnn -lcurand -lnvToolsExt

# OpenBLAS
LIBS += -L\$(BLAS_PATH)/lib -lopenblas

# NCCL
ifneq (\$(NCCL_PATH),)
	LIBS += -L\$(NCCL_PATH)/lib -lnccl
else
	LIBS += -lnccl
endif

# PTX code generation
#
GENCODE :=
SMS := 30 35 37 50 52
# With CUDA 9.0 and higher, 20 is no longer supported
# SMS := 20 30 35 37 50 52

ifeq (\$(GENCODE),)
\$(foreach sm,\$(SMS),\$(eval GENCODE += -gencode arch=compute_\$(sm),code=sm_\$(sm)))
SMMAX := \$(lastword \$(sort \$(SMS)))
ifneq (\$(SMMAX),)
	GENCODE += -gencode arch=compute_\$(SMMAX),code=compute_\$(SMMAX)
endif
endif

OBJS := executioncontext.o timer.o threadsafequeue.o waitfreequeue.o thetaqueue.o memorymanager.o list.o bytebuffer.o bufferpool.o arraylist.o stream.o kernel.o operator.o operatordependency.o dataflow.o variableschema.o variable.o localvariable.o kernelconfigurationparameter.o kernelscalar.o model.o modelmanager.o resulthandler.o databuffer.o kernelmap.o batch.o callbackhandler.o taskhandler.o solverconfiguration.o measurementlist.o device.o lightweightdatasethandler.o recorddataset.o doublebuffer.o synch/common.o synch/default.o synch/downpour.o synch/eamsgd.o synch/hogwild.o synch/polyakruppert.o synch/sma.o synch/synchronouseamsgd.o synch/synchronoussgd.o cudnn/cudnntensor.o cudnn/cudnnconvparams.o cudnn/cudnnpoolparams.o cudnn/cudnnreluparams.o cudnn/cudnnsoftmaxparams.o cudnn/cudnnbatchnormparams.o cudnn/cudnndropoutparams.o cudnn/cudnnhelper.o
KNLS := kernels/classify.o kernels/accuracy.o kernels/gradientdescentoptimiser.o kernels/innerproduct.o kernels/innerproductgradient.o kernels/matmul.o kernels/noop.o kernels/noopstateless.o kernels/softmax.o kernels/softmaxgradient.o kernels/softmaxloss.o kernels/softmaxlossgradient.o kernels/pool.o kernels/poolgradient.o kernels/relu.o kernels/relugradient.o kernels/conv.o kernels/convgradient.o kernels/dropout.o kernels/dropoutgradient.o kernels/lrn.o kernels/lrngradient.o kernels/matfact.o kernels/cudnnconv.o kernels/cudnnconvgradient.o kernels/cudnnpool.o kernels/cudnnpoolgradient.o kernels/cudnnrelu.o kernels/cudnnrelugradient.o kernels/cudnnsoftmax.o kernels/cudnnsoftmaxgradient.o kernels/datatransform.o kernels/batchnorm.o kernels/batchnormgradient.o kernels/cudnnbatchnorm.o kernels/cudnnbatchnormgradient.o kernels/cudnndropout.o kernels/cudnndropoutgradient.o kernels/elementwiseop.o kernels/elementwiseopgradient.o kernels/concat.o kernels/concatgradient.o kernels/sleep.o kernels/optimisers/default.o kernels/optimisers/hogwild.o kernels/optimisers/downpour.o kernels/optimisers/eamsgd.o kernels/optimisers/synchronouseamsgd.o kernels/optimisers/synchronoussgd.o kernels/optimisers/sma.o kernels/optimisers/polyakruppert.o

CROSSBOWBASEINCLUDES := memorymanager.h debug.h utils.h

# TODO use \$(addprefix kernels/, \$(KNLS))

all: libCPU.so libdataset.so liblightweightdataset.so libGPU.so libBLAS.so libRNG.so librecords.so

libobjectref.so: objectref.o
	\$(NV) \$(LFL) -shared -o libobjectref.so objectref.o \$(LIBS)

objectref.o: objectref.c uk_ac_imperial_lsds_crossbow_device_ObjectRef.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@

uk_ac_imperial_lsds_crossbow_device_ObjectRef.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.ObjectRef
	
libCPU.so: CPU.o
	\$(NV) \$(LFL) -shared -o libCPU.so CPU.o \$(LIBS)
	
libGPU.so: GPU.o image/recordreader.o image/recordfile.o image/record.o image/image.o image/boundingbox.o image/rectangle.o image/yarng.o random/random.o random/generator.o \$(OBJS) \$(KNLS)
	\$(NV) \$(LFL) -shared -o libGPU.so GPU.o image/recordreader.o image/recordfile.o image/record.o image/image.o image/boundingbox.o image/rectangle.o image/yarng.o random/random.o random/generator.o \$(OBJS) \$(KNLS) \$(LIBS)
	
libBLAS.so: BLAS.o \$(OBJS) \$(KNLS)
	\$(NV) \$(LFL) -shared -o libBLAS.so BLAS.o \$(OBJS) \$(KNLS) \$(LIBS)

libRNG.so: random/random.o random/generator.o
	\$(CPP) -W -Wall -DWARNING -fPIC -Wno-unused-function -shared -o libRNG.so random/random.o random/generator.o 

libdataset.so: dataset.o datasetfilemanager.o datasetfilehandler.o datasetfile.o memoryregistry.o memoryregion.o memoryregionpool.o \$(OBJS) \$(KNLS)
	\$(NV) \$(LFL) -shared -o libdataset.so dataset.o datasetfilemanager.o datasetfilehandler.o datasetfile.o memoryregistry.o memoryregion.o memoryregionpool.o \$(OBJS) \$(KNLS) \$(LIBS)

liblightweightdataset.so: lightweightdataset.o lightweightdatasetmanager.o lightweightdatasetprocessor.o datasetfile.o memoryregistry.o lightweightdatasetbuffer.o \$(OBJS) \$(KNLS)
	\$(NV) \$(LFL) -shared -o liblightweightdataset.so lightweightdataset.o lightweightdatasetmanager.o lightweightdatasetprocessor.o datasetfile.o memoryregistry.o lightweightdatasetbuffer.o \$(OBJS) \$(KNLS) \$(LIBS)

librecords.so: image/recordreader.o image/recordfile.o image/record.o image/image.o image/boundingbox.o image/rectangle.o image/yarng.o \$(OBJS) \$(KNLS)
	\$(NV) \$(LFL) -shared -o librecords.so image/recordreader.o image/recordfile.o image/record.o image/image.o image/boundingbox.o image/rectangle.o image/yarng.o \$(OBJS) \$(KNLS) \$(LIBS)
	
CPU.o: CPU.c uk_ac_imperial_lsds_crossbow_device_TheCPU.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
BLAS.o: BLAS.c uk_ac_imperial_lsds_crossbow_device_blas_BLAS.h BLAS.h bufferpool.h bytebuffer.h debug.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

GPU.o: GPU.c uk_ac_imperial_lsds_crossbow_device_TheGPU.h executioncontext.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

random/random.o: random/random.cpp random/generator.hpp uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator.h
	\$(CPP) \$(INCLUDES) -W -Wall -DWARNING -fPIC -Wno-unused-function -c \$< -o \$@
	
random/generator.o: random/generator.cpp random/generator.hpp
	\$(CPP) \$(INCLUDES) -W -Wall -DWARNING -fPIC -Wno-unused-function -c \$< -o \$@
	
dataset.o: dataset.c uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

datasetfilemanager.o: datasetfilemanager.c datasetfilemanager.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
datasetfilehandler.o: datasetfilehandler.c datasetfilehandler.h list.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

datasetfile.o: datasetfile.c datasetfile.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
memoryregionpool.o: memoryregionpool.c memoryregionpool.h memoryregion.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
memoryregion.o: memoryregion.c memoryregion.h datasetfile.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

memoryregistry.o: memoryregistry.c memoryregistry.h memoryregistrynode.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
lightweightdataset.o: lightweightdataset.c uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

lightweightdatasetmanager.o: lightweightdatasetmanager.c lightweightdatasetmanager.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
lightweightdatasethandler.o: lightweightdatasethandler.c lightweightdatasethandler.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

lightweightdatasetprocessor.o: lightweightdatasetprocessor.c lightweightdatasetprocessor.h list.h lightweightdatasettask.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
lightweightdatasetbuffer.o: lightweightdatasetbuffer.c lightweightdatasetbuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

recorddataset.o: recorddataset.c recorddataset.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

doublebuffer.o: doublebuffer.c doublebuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

# === [Helpers for SGD (cross-replica synchronisation variants)] ===
#

synch/common.o: synch/common.c synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@

synch/default.o: synch/default.c synch/default.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/downpour.o: synch/downpour.c synch/downpour.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/eamsgd.o: synch/eamsgd.c synch/eamsgd.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/hogwild.o: synch/hogwild.c synch/hogwild.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/polyakruppert.o: synch/polyakruppert.c synch/polyakruppert.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/sma.o: synch/sma.c synch/sma.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/synchronouseamsgd.o: synch/synchronouseamsgd.c synch/synchronouseamsgd.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
synch/synchronoussgd.o: synch/synchronoussgd.c synch/synchronoussgd.h synch/common.h executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@

# === [End of SGD helpers] ===

image/recordreader.o: image/recordreader.c image/recordreader.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

image/recordfile.o: image/recordfile.c image/recordfile.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

image/record.o: image/record.c image/record.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

image/image.o: image/image.c image/image.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

image/boundingbox.o: image/boundingbox.c image/boundingbox.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
image/rectangle.o: image/rectangle.c image/rectangle.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

image/yarng.o: image/yarng.cpp image/yarng.h random/generator.hpp random/generator.o
	\$(CPP) \$(INCLUDES) -W -Wall -DWARNING -fPIC -Wno-unused-function -c \$< -o \$@

uk_ac_imperial_lsds_crossbow_device_TheCPU.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.TheCPU

uk_ac_imperial_lsds_crossbow_device_TheGPU.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.TheGPU

uk_ac_imperial_lsds_crossbow_device_blas_BLAS.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.blas.BLAS

uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.random.RandomGenerator

uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.dataset.DatasetMemoryManager

uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager.h:
	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.dataset.LightWeightDatasetMemoryManager

executioncontext.o: executioncontext.c executioncontext.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
timer.o: timer.c timer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
bytebuffer.o: bytebuffer.c bytebuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
bufferpool.o: bufferpool.c bufferpool.h bytebuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

list.o: list.c list.h listnode.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
		
arraylist.o: arraylist.c arraylist.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

stream.o: stream.c stream.h databuffer.h variable.h list.h dataflow.h operator.h model.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernel.o: kernel.c kernel.h variable.h arraylist.h threadsafequeue.h localvariable.h kernelconfigurationparameter.h kernelscalar.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
operator.o: operator.c operator.h kernel.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

operatordependency.o: operatordependency.c operatordependency.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
dataflow.o: dataflow.c dataflow.h operator.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

variableschema.o: variableschema.c variableschema.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
		
variable.o: variable.c variable.h variableschema.h databuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

model.o: model.c model.h variable.h databuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

modelmanager.o: modelmanager.c modelmanager.h model.h threadsafequeue.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
memorymanager.o: memorymanager.c memorymanager.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
threadsafequeue.o: threadsafequeue.c threadsafequeue.h listnode.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

waitfreequeue.o: waitfreequeue.c waitfreequeue.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
thetaqueue.o: thetaqueue.c thetaqueue.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

resulthandler.o: resulthandler.c resulthandler.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
databuffer.o: databuffer.c databuffer.h threadsafequeue.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernelmap.o: kernelmap.c kernelmap.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
batch.o: batch.c batch.h variableschema.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
localvariable.o: localvariable.c localvariable.h variable.h threadsafequeue.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernelconfigurationparameter.o: kernelconfigurationparameter.c kernelconfigurationparameter.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernelscalar.o: kernelscalar.c kernelscalar.h databuffer.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

callbackhandler.o: callbackhandler.c callbackhandler.h list.h threadsafequeue.h modelmanager.h resulthandler.h stream.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

taskhandler.o: taskhandler.c taskhandler.h list.h threadsafequeue.h callbackhandler.h stream.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

solverconfiguration.o: solverconfiguration.c solverconfiguration.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
measurementlist.o: measurementlist.c measurementlist.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
device.o: device.c device.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

# === [cuDNN kernel compilation] ===
#
cudnn/cudnnhelper.o: cudnn/cudnnhelper.c cudnn/cudnnhelper.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

cudnn/cudnntensor.o: cudnn/cudnntensor.c cudnn/cudnntensor.h \$(CROSSBOWBASEINCLUDES)
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

cudnn/cudnnconvparams.o: cudnn/cudnnconvparams.c cudnn/cudnnconvparams.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
cudnn/cudnnpoolparams.o: cudnn/cudnnpoolparams.c cudnn/cudnnpoolparams.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

cudnn/cudnnreluparams.o: cudnn/cudnnreluparams.c cudnn/cudnnreluparams.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
cudnn/cudnnsoftmaxparams.o: cudnn/cudnnsoftmaxparams.c cudnn/cudnnsoftmaxparams.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

cudnn/cudnnbatchnormparams.o: cudnn/cudnnbatchnormparams.c cudnn/cudnnbatchnormparams.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
cudnn/cudnndropoutparams.o: cudnn/cudnndropoutparams.c cudnn/cudnndropoutparams.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/cudnnconv.o: kernels/cudnnconv.cu kernels/cudnnconv.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnconvgradient.o: kernels/cudnnconvgradient.cu kernels/cudnnconvgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnpool.o: kernels/cudnnpool.cu kernels/cudnnpool.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnpoolgradient.o: kernels/cudnnpoolgradient.cu kernels/cudnnpoolgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnrelu.o: kernels/cudnnrelu.cu kernels/cudnnrelu.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnrelugradient.o: kernels/cudnnrelugradient.cu kernels/cudnnrelugradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnsoftmax.o: kernels/cudnnsoftmax.cu kernels/cudnnsoftmax.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnsoftmaxgradient.o: kernels/cudnnsoftmaxgradient.cu kernels/cudnnsoftmaxgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnbatchnorm.o: kernels/cudnnbatchnorm.cu kernels/cudnnbatchnorm.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnnbatchnormgradient.o: kernels/cudnnbatchnormgradient.cu kernels/cudnnbatchnormgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/cudnndropout.o: kernels/cudnndropout.cu kernels/cudnndropout.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/cudnndropoutgradient.o: kernels/cudnndropoutgradient.cu kernels/cudnndropoutgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

# === [End of cuDNN kernel compilation] ===
	
# === [Kernel compilation] ===
#

kernels/accuracy.o: kernels/accuracy.cu kernels/accuracy.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/classify.o: kernels/classify.cu kernels/classify.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/gradientdescentoptimiser.o: kernels/gradientdescentoptimiser.cu kernels/gradientdescentoptimiser.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

# === [Helpers for SGD (per replica sychronisation variants)] ===

kernels/optimisers/default.o: kernels/optimisers/default.cu kernels/optimisers/default.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
kernels/optimisers/hogwild.o: kernels/optimisers/hogwild.cu kernels/optimisers/hogwild.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@

kernels/optimisers/downpour.o: kernels/optimisers/downpour.cu kernels/optimisers/downpour.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
kernels/optimisers/eamsgd.o: kernels/optimisers/eamsgd.cu kernels/optimisers/eamsgd.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
kernels/optimisers/synchronouseamsgd.o: kernels/optimisers/synchronouseamsgd.cu kernels/optimisers/synchronouseamsgd.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
kernels/optimisers/synchronoussgd.o: kernels/optimisers/synchronoussgd.cu kernels/optimisers/synchronoussgd.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@

kernels/optimisers/sma.o: kernels/optimisers/sma.cu kernels/optimisers/sma.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
kernels/optimisers/polyakruppert.o: kernels/optimisers/polyakruppert.cu kernels/optimisers/polyakruppert.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c $< -o \$@
    
# === [End of SGD helpers] ===

kernels/innerproduct.o: kernels/innerproduct.cu kernels/innerproduct.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/innerproductgradient.o: kernels/innerproductgradient.cu kernels/innerproductgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/matmul.o: kernels/matmul.cu kernels/matmul.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/noop.o: kernels/noop.cu kernels/noop.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/noopstateless.o: kernels/noopstateless.cu kernels/noopstateless.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/softmax.o: kernels/softmax.cu kernels/softmax.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/softmaxgradient.o: kernels/softmaxgradient.cu kernels/softmaxgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/softmaxloss.o: kernels/softmaxloss.cu kernels/softmaxloss.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/softmaxlossgradient.o: kernels/softmaxlossgradient.cu kernels/softmaxlossgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/conv.o: kernels/conv.cu kernels/conv.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/convgradient.o: kernels/convgradient.cu kernels/convgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/pool.o: kernels/pool.cu kernels/pool.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/poolgradient.o: kernels/poolgradient.cu kernels/poolgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/relu.o: kernels/relu.cu kernels/relu.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/relugradient.o: kernels/relugradient.cu kernels/relugradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/dropout.o: kernels/dropout.cu kernels/dropout.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/dropoutgradient.o: kernels/dropoutgradient.cu kernels/dropoutgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/lrn.o: kernels/lrn.cu kernels/lrn.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/lrngradient.o: kernels/lrngradient.cu kernels/lrngradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/batchnorm.o: kernels/batchnorm.cu kernels/batchnorm.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/batchnormgradient.o: kernels/batchnormgradient.cu kernels/batchnormgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/elementwiseop.o: kernels/elementwiseop.cu kernels/elementwiseop.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/elementwiseopgradient.o: kernels/elementwiseopgradient.cu kernels/elementwiseopgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/concat.o: kernels/concat.cu kernels/concat.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/concatgradient.o: kernels/concatgradient.cu kernels/concatgradient.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/matfact.o: kernels/matfact.cu kernels/matfact.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
kernels/datatransform.o: kernels/datatransform.cu kernels/datatransform.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@

kernels/sleep.o: kernels/sleep.cu kernels/sleep.h
	\$(NV) \$(INCLUDES) \$(LFL) \$(GENCODE) -c \$< -o \$@
	
# === [End of kernel compilation] ===
	
test: image/testrecordreader.c image/testbatchreader.c testrecorddataset.c
	\$(NV) \$(INCLUDES) \$(LFL) image/testrecordreader.c -o image/testrecordreader -L\$(CBOW_PATH)/clib-multigpu -lGPU -lCPU -lBLAS -lRNG -lrecords \$(LIBS)
	\$(NV) \$(INCLUDES) \$(LFL) image/testbatchreader.c  -o image/testbatchreader  -L\$(CBOW_PATH)/clib-multigpu -lGPU -lCPU -lBLAS -lRNG -lrecords \$(LIBS)
	\$(NV) \$(INCLUDES) \$(LFL) testrecorddataset.c  -o testrecorddataset  -L\$(CBOW_PATH)/clib-multigpu -lGPU -lCPU -lBLAS -lRNG -lrecords \$(LIBS)
	
clean:
	rm -f *.o *.so
	rm -f kernels/*.o
	rm -f cudnn/*.o
	rm -f random/*.o
	rm -f image/*.o
	rm -rf *.dSYM
	rm -f test
	rm -f image/testrecordreader
	rm -f image/testbatchreader
	rm -f testrecorddataset

!endoftemplate!

exit 0
