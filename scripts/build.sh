#!/bin/sh

# Check if CROSSBOW_HOME is set
if [ -z "$CROSSBOW_HOME" ]; then
    echo "error: \$CROSSBOW_HOME is not set"
    exit 1
fi

# Source common functions
. "$CROSSBOW_HOME"/scripts/common.sh

# Source configuration parameters
. "$CROSSBOW_HOME"/scripts/crossbow.conf

[ ! -d "$CROSSBOW_LIBDIR" ] && mkdir "$CROSSBOW_LIBDIR"

# Goto directory where pom.xml exists
cd $POM_DIR

crossbowFileExistsOrExit "$POM"

TRANSCRIPT="build.log"

[ -f $TRANSCRIPT ] && rm -f $TRANSCRIPT

crossbowProgramExistsOrExit "mvn"

mvn package -q -e -X >$TRANSCRIPT 2>&1
[ $? -ne 0 ] && {
        echo "error: Crossbow Java library compilation failed (transcript written on $POM_DIR/$TRANSCRIPT)"
        exit 1
}

# Crossbow was successfully build
echo "Crossbow Java library build successful (transcript written on $POM_DIR/$TRANSCRIPT)"

JAR="$CROSSBOW_HOME/target/crossbow-0.0.1-SNAPSHOT.jar"
crossbowFileExistsOrExit "$JAR"

# Copy JAR to lib folder
[ -f "$JAR" ] && cp "$JAR" "$CROSSBOW_LIBDIR"

LIBFILE="$CROSSBOW_LIBDIR/crossbow-0.0.1-SNAPSHOT.jar"
crossbowFileExistsOrExit "$LIBFILE"

jarsize=`wc -c < "$LIBFILE" | sed -e 's/^[ \t]*//'`
echo "Output written on $LIBFILE ($jarsize bytes)"

#
# Building Crossbow C libraries
#

crossbowProgramExistsOrExit "make"
crossbowProgramExistsOrExit "gcc"

# Goto clib/ directory
[ ! -d "$CLIB_DIR" ] && mkdir "$CLIB_DIR"
cd $CLIB_DIR

./genmakefile.sh

# Log output to clib/build.log
[ -f $TRANSCRIPT ] && rm -f $TRANSCRIPT

# Clean-up
make clean >>$TRANSCRIPT 2>&1

# Make libraries
make -j >>$TRANSCRIPT 2>&1
[ $? -ne 0 ] && {
echo "error: Crossbow C libraries compilation failed (transcript written on $CLIB_DIR/$TRANSCRIPT)"
exit 1
}

CPULIB="$CLIB_DIR"/libCPU.so
GPULIB="$CLIB_DIR"/libGPU.so
BLASLIB="$CLIB_DIR"/libBLAS.so
RNGLIB="$CLIB_DIR"/libRNG.so
DATALIB="$CLIB_DIR"/libdataset.so

for lib in $CPULIB $GPULIB $BLASLIB $RNGLIB $DATALIB
do
  crossbowFileExistsOrExit "$lib"
  libsize=`wc -c < "$lib" | sed -e 's/^[ \t]*//'`
  echo "Output written on $lib ($libsize bytes)"
done

echo "Crossbow C libraries build successful (transcript written on $CLIB_DIR/$TRANSCRIPT)"

# Create Crossbow directories: .pids, logs
cd $CROSSBOW_HOME
[ ! -d "$CROSSBOW_LOGDIR" ] && mkdir "$CROSSBOW_LOGDIR"
[ ! -d "$CROSSBOW_PIDDIR" ] && mkdir "$CROSSBOW_PIDDIR"

exit 0
