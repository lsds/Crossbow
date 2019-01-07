#!/bin/bash

# Check if CROSSBOW_HOME is set
if [ -z "$CROSSBOW_HOME" ]; then
        echo "error: \$CROSSBOW_HOME is not set"
        exit 1
fi

# Source common functions
. "$CROSSBOW_HOME"/scripts/common.sh

# Source configuration parameters
. "$CROSSBOW_HOME"/scripts/crossbow.conf

USAGE="usage: run.sh [ --alias ] [ --mode {foreground, background} ] [ --class ] [ -- <class arguments> ]"

# Note: only one of the two variables below should be set to true
NVPROF=false
MEMCHK=false

# Parse pom.xml to extract JAR paths

POM_JARS=
parsePomJars () {
	cd $CROSSBOW_HOME

  	crossbowDirectoryExistsOrExit "$MVN_REPOSITORY"
  	crossbowFileExistsOrExit "$POM"

  	POM_JARS=$(eval "sed `sed -n -e "/<properties>/,/<\/properties>/s/ *<\([^>]*\)>\([^<]*\)<.*$/-e 's\/\\\\$\\{\1\\}\/\2\/' /p" $POM |
  	tr -d '\n'` $POM" |
  	sed -n \
    		-e "/<exclusions>/,/<\/exclusions>/d" \
    		-e "/<groupId>/s/\./\//g" \
    		-e "/<dependencies>/,/<\/dependencies>/p" $POM |
    	tr -d "\n\t " |
    	sed \
    		-e "s/<scope>[^<]*<\/scope>//g" \
    		-e "s/<!--//g" \
    		-e "s/-->//g" \
    		-e "s/<dependencies>//g" \
    		-e "s/<\/dependencies>//g" \
    		-e "s/<systemPath>//g" \
    		-e "s/<\/systemPath>//g" \
    		-e "s|<dependency><groupId>\([^<]*\)<\/groupId><artifactId>\([^<]*\)<\/artifactId><version>\([^<]*\)<\/version><\/dependency>|${MVN_REPOSITORY}\/\1\/\2\/\3\/\2-\3.jar:|g" |
    	sed \
    		-e "s/<groupId>.*<\/groupId>//" \
    		-e "s/<version>.*<\/version>//" \
    		-e "s/<artifactId>.*<\/artifactId>//" \
    		-e "s/<dependency>//g" \
    		-e "s/<\/dependency>//g" \
    		-e "s/<groupId>//g" \
   		-e "s/<\/groupId>//g" \
    		-e "s/<version>//g" \
    		-e "s/<\/version>//g" \
    		-e "s/<artifactId>//g" \
    		-e "s/<\/artifactId>//g")
}

#
# Main
#
# Command-line arguments

MODE="foreground"
# When mode is `background`, alias must be set
ALIAS=
# The Crossbow application
CLS=
# The Crossbow application arguments
ARGS=

# Parse command-line arguments

while :
do
        case "$1" in
                -m | --mode)
                crossbowOptionInSet "$1" "$2" "foreground" "background" || exit 1
                MODE="$2"
                # if [ \( "$MODE" != "foreground" \) -a \( "$MODE" != "background" \) ]; then
                #       echo "error: invalid mode: $MODE"
                #       exit 1
                # fi 
                shift 2
                ;;
                -a | --alias)
                crossbowOptionIsAlpha "$1" "$2" || exit 1
                ALIAS="${2}-$(date +%Y-%m-%m-%H:%M)"
                shift 2
                ;;
                -c | --class)
                CLS="$2"
                # Check class exists
                crossbowFileExistsOrExit "$TESTS`echo ${CLS} | tr '.' '/'`.class"
                shift 2
                ;;
                -h | --help)
                echo $USAGE
                exit 0
                ;;
                --) # End of all options
                shift
                break
                ;;
                -*)
                echo "error: invalid option: $1" >&2
                exit 1
                ;;
                *) # done, if string is empty
                if [ -n "$1" ]; then
                        echo "error: invalid argument: $1"
                        exit 1
                fi
                break
                ;;
        esac
done

# The remaining arguments are class arguments
ARGS="$@"

# Check that CLS is set (if set, it is correct)
[ -z "$CLS" ] && {
        echo "error: no class specified"
        exit 1
}

# Set Java classpath variable, $JCP
parsePomJars

crossbowFileExistsOrExit $CROSSBOW
crossbowDirectoryExistsOrExit $TESTS

JCP=".:${CROSSBOW}:${POM_JARS}:${TESTS}"
echo -e "\nClasspath contains:\n`echo $JCP | tr ':' '\n'`"

#
# JVM options
#
OPTS="-server -XX:+UseConcMarkSweepGC"

OPTS="$OPTS -XX:NewRatio=${CROSSBOW_JVM_NEWRATIO}"
OPTS="$OPTS -XX:SurvivorRatio=${CROSSBOW_JVM_SURVIVORRATIO}"

OPTS="$OPTS -Xms${CROSSBOW_JVM_MS}g"
OPTS="$OPTS -Xmx${CROSSBOW_JVM_MX}g"

if [ "$CROSSBOW_JVM_LOGGC" = "true" ]; then
        # Log garbage collection events
        OPTS="$OPTS -Xloggc:gc.out"
fi

$CROSSBOW_RUN_LOG && crossbowLogRunCommand "java $OPTS -cp $JCP $CLS $ARGS"

errorcode=0

if [ "$MODE" = "foreground" ]; then
        if [ $NVPROF = true ]; then
                [ ! -d "$CROSSBOW_NVPROFDIR" ] && mkdir -p -- "$CROSSBOW_NVPROFDIR"
                nvprof -o "${CROSSBOW_NVPROFDIR}/${CLS##*.}-$(date +%Y-%m-%m-%H:%M).nvvp" java $OPTS -cp $JCP $CLS $ARGS
        elif [ $MEMCHK = true ]; then
                cuda-memcheck java $OPTS -cp $JCP $CLS $ARGS
        else
                java $OPTS -cp $JCP $CLS $ARGS
        fi
else
        # Running application in the background
        #
        # Check that class alias is set
        [ -z "$ALIAS" ] && {
                echo "error: class alias is not set"
                exit 1
        }
        #
        CMD="java $OPTS -cp $JCP $CLS $ARGS"
        # Try trap signals
        crossbowSignalTrapped || crossbowProcessTrap
        crossbowProcessStart $ALIAS $CMD
        #
        $CROSSBOW_VERBOSE && echo "Running application \"$ALIAS\"..."
        interrupted=0

        while true; do
                read -n 1 -s -t 1
                key=$?
                if [ $key -eq 0 ]; then
                        interrupted=1
                        # Set error code
                        errorcode=1
                        break
                fi
                crossbowProcessIsRunning $ALIAS $CMD || {
                        echo "" # Line break
                        echo "error: application \"$ALIAS\" has failed (check "$CROSSBOW_LOGDIR"/$ALIAS.err for errors)"
                        # Set error code
                        errorcode=1
                        break
                }
        done
        if [ $interrupted -ne 0 ]; then
                echo "Interrupted"
        fi
        # 
        # Check until Crossbow measurements are flushed to output file
        # only if we have exited gracefully
        if [ $errorcode -eq 0 ]; then
                echo ""
                crossbowProcessDone $ALIAS
                if [ $? -ne 0 ]; then
                        echo "warning: failed to detect whether measurements have been flushed"
                        errorcode=1
                fi
        fi
        # Stop the process (and clean-up, even if there has been an error)
        crossbowProcessStop $ALIAS $CMD
        # echo "Done"
fi

exit $errorcode


