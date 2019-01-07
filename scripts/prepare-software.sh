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

USAGE="usage: prepare-software.sh [ -f | --force-essential ] [ -s | --skip-optional ]"
# Command-line arguments

# Skips the installation of optional packages
SKIP=0

# Forces the installation of essential packages
FORCE=0

# Parse command-line arguments
while :
do
	case "$1" in
	-f | --force-essential)
	FORCE=1
	shift 1
	;;
	-s | --skip-optional)
	SKIP=1
	shift 1
	;;
	-h | --help)
	;;
	--)
	shift
	break
	;;
	-*)
	echo "error: invalid option: $1" >&2
	exit 1
	;;
	*)
	if [ -n "$1" ]; then
	echo "error: invalid argument: $1"
	exit 1
	fi
	break
	;;
	esac
done

F="$CROSSBOW_HOME/scripts/crossbow.prerequisites"
crossbowFileExistsOrExit "$F"

LOG="$CROSSBOW_HOME/apt.log"
# Rotate previous apt.log file
crossbowRotateAptLog

# (Re)synchronize package index files
crossbowProgramExists "apt-get" && {
	
	# If apt-get exists, let's initialise the apt.log file
	echo "#" >"$LOG"
	echo "# Crossbow's apt install log" >>"$LOG"
	echo "# Generated on `uname -n` running `uname -s` on `date`" >>"$LOG"
	echo "#" >>"$LOG"
	
	echo "Updating package list. This may take some time."
	sudo apt-get update >>"$LOG" 2>&1
}

# Parse prerequisites, skipping comments
sed -e '/#.*$/d' -e '/^$/d' "$F" | \
(\
while read line; do
	
	CMD=`echo $line | awk '{ split($0, t, ":"); print t[1] }'`
	PKG=`echo $line | awk '{ split($0, t, ":"); print t[2] }'`
	FLG=`echo $line | awk '{ n = split($0, t, ":"); if (n > 2) print t[3]; else print "" }'`
	
	# Trim whitespaces
	CMD=`echo $CMD | sed 's/[ 	]*//'` # <space> or <tab>
	PKG=`echo $PKG | sed 's/[ 	]*//'`
	FLG=`echo $FLG | sed 's/[ 	]*//'`
	
	# Check if strings are correct, using pattern X
	XPR='[\. ]'
	[ -n "$CMD" ] && {
		if [[ "$CMD" =~ $XPR ]]; then
			echo "error: invalid command: \"$CMD\""
			exit 1
		fi
	}
	[ -n "$PKG" ] && {
		if [[ "$PKG" =~ $XPR ]]; then
			echo "error: invalid package: \"$PKG\""
			exit 1
		fi
	}
	[ -n "$FLG" ] && {
		# Check if flag is a valid string
		if [ \( "$FLG" != "y" \) -a \( "$FLG" != "n" \) ]; then
			echo "error: invalid flag: \"$FLG\""
			exit 1
		fi
	} || FLG="y" # The default flag is yes
	
	# Is command set?
	[ -n "$CMD" ] && {
		
		# Does the command exist?
		crossbowProgramExists $CMD
		[ $? -eq 0 ] && {
			
			# Get command version
			VER=`crossbowProgramVersion $CMD`
			printf "%-10s %s\n" "$CMD" "$VER"
		
		} || {  # Command does not exist
			lvl=
			[ "$FLG" = "y" ] && lvl="error" || lvl="warning"
			echo "$lvl: $CMD: command not found"
			# Is a package set?
			[ -n "$PKG" ] && {
				
				crossbowAskToInstall $PKG $FLG $SKIP $FORCE </dev/tty # Input from keyboard
				answer=$?
				
				# Does the user agree?
				case $answer in
				0)
				crossbowInstallPackage "$PKG" "$LOG" || exit 1
				;;
				1)
				exit 2
				;;
				*) ;; # Go to next line
				esac
				
			} || { 
				# Package was not set. Is it essential?
				[ "$FLG" = "y" ] && exit 1
			}
		}
	} || {  # Check if only a package was specified
		
		[ -n "$PKG" ] && {
			
			crossbowAskToInstall $PKG $FLG $SKIP $FORCE </dev/tty # Input from keyboard
			answer=$?
			
			# Does the user agree?
			case $answer in
			0)
			crossbowInstallPackage "$PKG" "$LOG" || exit 1
			;;
			1)
			exit 2
			;;
			*) ;; # Go to next line
			esac
		}
	}
done) || {
	errorcode=$?
	# "while read" loop failed or aborted
	crossbowProgramExists "apt-get" && {
	err=""
	[ $errorcode -eq 1 ] && err="failed" || err="aborted"
	echo "error: apt install $err (transcript written on $LOG)"
	}
	exit 1 
} 

crossbowProgramExists "apt-get" && {
# Apt install was successful
echo "Apt install successful (transcript written on $LOG)"
}

# Check libraries: Is CUDA_HOME set?

# Compile & build
$CROSSBOW_HOME/scripts/build.sh
[ $? -ne 0 ] && exit 1

# Create Crossbow directories: .pids, logs
[ ! -d "$CROSSBOW_LOGDIR" ] && mkdir "$CROSSBOW_LOGDIR"
[ ! -d "$CROSSBOW_PIDDIR" ] && mkdir "$CROSSBOW_PIDDIR"

exit 0
