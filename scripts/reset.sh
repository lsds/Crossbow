#!/bin/bash
#
# Delete all files generated automatically
# by various scripts in this distribution.
#
# usage: ./reset.sh
#

# Check if CROSSBOW_HOME is set
if [ -z "$CROSSBOW_HOME" ]; then
	echo "error: \$CROSSBOW_HOME is not set"
	exit 1
fi

# Source common functions
. "$CROSSBOW_HOME"/scripts/common.sh

# Source configuration parameters
. "$CROSSBOW_HOME"/scripts/crossbow.conf

echo -n "Delete all generated files, including any reproduced results (yes/no)? "
while true; do
	read a
	case "$a" in
	y|yes)
	result=0
	break
	;;
	n|no)
	result=1
	break
	;;
	*)
	echo -n "Invalid option: \"$a\". Choose 'yes' or 'no': "
	;;
	esac
done

[ $result -eq 1 ] && exit 1

# Start

rm -rf "$CROSSBOW_LOGDIR"
rm -rf "$CROSSBOW_PIDDIR"

rm -f "$CROSSBOW_TRAP"

# Delete log files in Crossbow home directory

rm -f "$CROSSBOW_HOME"/apt.log*
rm -f "$CROSSBOW_HOME"/build.log
rm -f "$CROSSBOW_HOME"/run.log

# Delete Crossbow code distribution

rm -rf "$CROSSBOW_HOME"/target
rm -rf "$CROSSBOW_HOME"/lib

# clib 
rm -f "$CROSSBOW_HOME"/clib-multigpu/Makefile*
rm -f "$CROSSBOW_HOME"/clib-multigpu/build.log
rm -f "$CROSSBOW_HOME"/clib-multigpu/*.so
rm -f "$CROSSBOW_HOME"/clib-multigpu/*.o

# Delete CUDA files
# TBD

# Delete intermediate files in doc/
# (equivalent to `make clean`)
# TBD

# Delete reproduced results in doc/
# TBD

# Delete any reproduced results for each figure
# TBD

echo "Bye."

exit 0
