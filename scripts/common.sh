#!/bin/bash

# Check if CROSSBOW_HOME is set
if [ -z $CROSSBOW_HOME ]; then
    echo "error: \$CROSSBOW_HOME is not set"
    exit 1
fi

# Source configuration parameters
. $CROSSBOW_HOME/scripts/crossbow.conf

crossbowProcessStart () {
	crossbowDirectoryExistsOrExit "$CROSSBOW_LOGDIR"
	crossbowDirectoryExistsOrExit "$CROSSBOW_PIDDIR"
	#
	name=$1
	crossbowProcessClear "$@"
	crossbowProcessIsRunning "$@"
	[ $? -eq 0 ] && return 1
	shift 1
	(
		# Redirect standard file descriptors to log files
		[[ -t 0 ]] && exec 0</dev/null
		[[ -t 1 ]] && exec 1>"$CROSSBOW_LOGDIR/${name}.out"
		[[ -t 2 ]] && exec 2>"$CROSSBOW_LOGDIR/${name}.err"
		# Close non-standard file descriptors
		eval exec {3..255}\>\&-
		trap '' 1 2 # Ignore HUP INT in child process
		exec "$@"
	) &
	pid=$!
	disown -h $pid
	$CROSSBOW_VERBOSE && echo "$name's pid is $pid"
	echo $pid > "$CROSSBOW_PIDDIR/$name.pid"
	return 0
}

crossbowProcessStop () {
	name=$1
	[ -s "$CROSSBOW_PIDDIR/$name.pid" ] && (
		pid=`cat "$CROSSBOW_PIDDIR/$name.pid"`
		kill -15 $pid >/dev/null 2>&1
		rm "$CROSSBOW_PIDDIR/$name.pid"
		return 0
	) || (
		echo "error: $CROSSBOW_PIDDIR/$name.pid not found"
		return 1
	)
}

crossbowProcessIsRunning () {
	name=$1
	# Check if process $name is running
	[ -s "$CROSSBOW_PIDDIR/$name.pid" ] && (
		# $CROSSBOW_VERBOSE && echo "$name.pid found"
		pid=`cat "$CROSSBOW_PIDDIR/$name.pid"`
		ps -p $pid >/dev/null 2>&1
		return $?
	) || ( 
		# unlikely
		shift 1
		t=\""$@"\"
		pgrep -lf "$t" >/dev/null 2>&1
		[ $? -eq 1 ] && return 1 || (
			echo "warning: $name is beyond our control"
			return 0
		)
	)
}

crossbowProcessClear () {
	name=$1
	shift 1
	# Check if $name.pid exists but process is not running
	t=\""$@"\"
	pgrep -lf "$t" >/dev/null 2>&1
	if [ \( $? -eq 1 \) -a \( -f "$PIDDIR/$name.pid" \) ]; then
		rm "$PIDDIR/$name.pid"
	fi
	return 0
}

crossbowProcessDone () {
	# 
	# Wait up to X seconds for measurements to be flushed in output file.
	#
	# The line to look for, printed by the PerformanceMonitor class, is:
	# 
	# [MON] Done.
	#
	name=$1
	
	F="$CROSSBOW_LOGDIR/${name}.out"
	
	[ ! -f "$F" ] && {
		echo "warning: \"$F\" not found"
		return 1
	}
	
	found=1 # Not found
	
	attempts=$CROSSBOW_WAITTIME
	length=${#attempts} # Length to control printf length
	while [ $attempts -gt 0 ]; do
		printf "\rWaiting up to %${length}ds for measurements to be flushed " $attempts
		let attempts--
		sleep 1
		cat "$F" | grep "\[MON\] Done." >/dev/null 2>&1
		if [ $? -eq 0 ]; then
			found=0
			break
		fi
	done
	echo "" # Line break
	return $found
}

crossbowSignalTrapped () {
	if [ -f "$CROSSBOW_TRAP" ]; then
		return 0
	else
		return 1
	fi
}

crossbowProcessTrap () {
	trap "crossbowProcessSignal" 1 2 3 8 11 16 17
}

crossbowProcessClearTrap () {
	trap - 1 2 3 8 11 16 17
	# Delete trap file, in exists
	rm -f "$CROSSBOW_TRAP"
}

crossbowProcessSignal () {
	echo "Signal received: shutting down..."
	crossbowShutdown
	# Delete trap file
	rm -f "$CROSSBOW_TRAP"
	exit 0
}

crossbowShutdown () {
	# Shutdown all running processes
	error=0
	for f in $(ls "$CROSSBOW_PIDDIR"/*.pid); do
		t=${f%.*}
		n=${t##*/}
		crossbowProcessStop $n
		if [ $? -ne 0 ]; then
			let error++
		fi
	done
	# Sanitise stdin
	stty sane
	return $error
}

crossbowProgramExists () {
	program=$1
	which $program >/dev/null 2>&1
	return $?
}

crossbowProgramExistsOrExit () {
	crossbowProgramExists "$1"
	if [ $? -ne 0 ]; then
		echo "error: $1: command not found"
		exit 1
	fi
}

crossbowProgramVersion () {
	version=""
	case "$1" in
	javac)
	version=`javac -version 2>&1 | awk '{ print $2 }'`
	;;
	java)
	version=`java -version 2>&1 | head -n 1 | awk -F '"' '{ print $2 }'`
	;;
	mvn)
	version=`mvn --version 2>&1 | head -n 1 | awk '{ print $3 }'`
	;;
	python)
	version=`python --version 2>&1 | awk '{ print $2 }'`
	;;
	make)
	version=`make --version 2>&1 | head -n 1 | awk '{ print $3 }'`
	;;
	gcc)
	version=`gcc -dumpversion 2>&1`
	;;
	perl)
	version=`perl -e 'print $^V;' 2>&1`
	;;
	latex)
	version=`latex -v 2>&1 | head -n 1`
	;;
	pdflatex)
	version=`pdflatex -v 2>&1 | head -n 1`
	;;
	bibtex)
	version=`bibtex -v 2>&1 | head -n 1 | awk '{ print $2 }'`
	;;
	epstopdf)
	version=`epstopdf -v 2>&1 | head -n 1 | awk '{ print $NF }'`
	;;
	gs)
	version=`gs -v 2>&1 | head -n 1`
	;;
	gnuplot)
	version=`gnuplot --version | awk '{ print $2 }'`
	;;
	*)
	version="(undefined version)"
	;;
	esac
	echo $version
}

crossbowSuperUser () {
	user=`id -u`
	[ $user != "0" ] && return 1
	return 0
}

crossbowAskToInstall () {
	P="$1"
	F="$2"
	
	OPT=$3 # Skip  installation of optional packages
	REQ=$4 # Force installation of required packages
	
	[ "$F" = "y" ] && [ $REQ -ne 0 ] && return 0 # User said  'yes'
	[ "$F" = "n" ] && [ $OPT -ne 0 ] && return 2 # User said 'skip'
	
	Q="Install package $1"
	[ "$F" = "y" ] && Q="$Q (yes/no)? " || Q="$Q (yes/no/skip)? "
	
	result=0
	echo -n "$Q"
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
		s|skip)
		[ $F = "n" ] && {
		result=2
		break
		} || {
		echo -n "Invalid option: \"$a\". Choose 'yes' or 'no': "
		}
		;;
		*)
		[ $F = "y" ] && {
		echo -n "Invalid option: \"$a\". Choose 'yes' or 'no': "
		} || {
		echo -n "Invalid option: \"$a\". Choose 'yes', 'no', or 'skip': "
		}
		;;
		esac
	done
	
	return $result
}

crossbowRotateAptLog () {
	next=0
	# Are there any rotated logs?
	ls "$CROSSBOW_HOME"/apt.log.* >/dev/null 2>&1
	if [ $? -eq 0 ]; then
		files=`ls "$CROSSBOW_HOME"/apt.log.*`
		# Find latest version of apt.log.*
		for file in $files; do
			curr=`echo $file | awk '{ n = split($0, t, "."); print t[n] }'`
			[ $curr -gt $next ] && next=$curr
		done
	fi
	# Is there an apt.log file? If yes, rotate it
	if [ -f "$CROSSBOW_HOME/apt.log" ]; then
		let next++
		mv "$CROSSBOW_HOME/apt.log" "$CROSSBOW_HOME/apt.log.$next"
	fi
}

crossbowInstallPackage () {
	
	crossbowProgramExists "apt-get" || return 1
	crossbowProgramExists    "dpkg" || return 1
	
	package="$1"
	logfile="$2"
	
	dpkg -s "$package" >/dev/null 2>&1
	# Is package already installed?
	[ $? -eq 0 ] && return 0
	
	if [ -f "$logfile" ]; then
	# Redirect output to apt.log (here, $logfile)
	sudo apt-get -y -q --allow-unauthenticated install "$package" >>"$logfile" 2>&1
	else
	sudo apt-get -y -q --allow-unauthenticated install "$package"
	fi
	
	# Is package installed?
	dpkg -s "$package" >/dev/null 2>&1
	if [ $? -eq 1 ]; then
		if [ -f "$logfile" ]; then
		echo "error: failed to install $package (transcript written on $logfile)"
		else
		echo "error: failed to install $package"
		fi
		return 1
	fi
	
	return 0
}

crossbowDirectoryExistsOrExit () {
	# Check if $1 is a directory
	if [ ! -d "$1" ]; then
		echo "error: $1: directory not found"
		exit 1
	fi
	return 0
}

crossbowFileExistsOrExit () {
	if [ ! -f "$1" ]; then
		echo "error: $1: file not found"
		exit 1
	fi
	return 0
}

crossbowFileExists () {
	V="$2"
	if [ ! -f "$1" ]; then
		[ "$V" = "-v" ] && echo "warning: $1: file not found"
		return 1
	fi
	return 0
}

crossbowFindJar () {
	find $1 -name *.jar
}

crossbowOptionInSet () {
	OPT="$1"
	ARG="$2"
	shift 2
	for VAL in $@
	do
		[ "$ARG" = "$VAL" ] && return 0
	done
	echo "error: invalid option: $OPT $ARG"
	return 1
}

crossbowOptionIsDuplicateFigureId () {
	OPT="$1"
	ARG="$2"
	FIG=`echo "$ARG" | tr -d [a-z,A-Z]`
	shift 2
	for VAL in $@
	do
		[ "$FIG" = "$VAL" ]	&& {
		echo "error: duplicate option: $OPT $ARG"
		return 1
		}
	done
	return 0
}

crossbowOptionIsPositiveInteger () {
	OPT="$1"
	ARG="$2"
	# Check that opt is a number
	NUMERIC='^[0-9]+$'
	if [[ ! "$ARG" =~ $NUMERIC ]]; then
		echo "error: invalid option: $OPT must be integer"
		return 1
	fi
	# Also check that it is > 0
	if [ $ARG -le 0 ]; then
		echo "error: invalid option: $OPT must be greater than 0"
		return 1
	fi
	return 0
}

crossbowOptionIsInteger () {
	OPT="$1"
	ARG="$2"
	# Check that opt is a number
	NUMERIC='^[0-9]+$'
	if [[ ! "$ARG" =~ $NUMERIC ]]; then
		echo "error: invalid option: $OPT must be integer"
		return 1
	fi
	# Also check that it is > 0
	if [ $ARG -lt 0 ]; then
		echo "error: invalid option: $OPT must be greater or equal to 0"
		return 1
	fi
	return 0
}

crossbowOptionIsIntegerWithinRange () {
	OPT="$1"
	ARG="$2"
	MIN="$3"
	MAX="$4"
	# Check that opt is a number
	NUMERIC='^[0-9]+$'
	if [[ ! "$ARG" =~ $NUMERIC ]]; then
		echo "error: invalid option: $OPT must be integer"
		return 1
	fi
	# Also check that it is within range
	if [ \( $ARG -lt $MIN \) -o \( $ARG -gt $MAX \) ]; then
		echo "error: invalid option: $OPT must be between $MIN and $MAX"
		return 1
	fi
	return 0
}

crossbowOptionIsBoolean () {
	OPT="$1"
	ARG="$2"
	if [ \( "$ARG" != "true" \) -a \( "$ARG" != "false" \) ]; then
		echo "error: invalid option: $OPT must be \"true\" or \"false\""
		return 1
	fi
	return 0
}

crossbowOptionIsAlpha () {
	OPT="$1"
	ARG="$2"
	ALPHA='^[a-zA-Z]+$'
	if [[ ! "$ARG" =~ $ALPHA ]]; then
		echo "error: invalid option: $OPT must contains only alphabetical chars"
		return 1
	fi
	return 0
}

crossbowOptionIsValidFigureId () {
	OPT="$1"
	ARG="$2"
	# TBD
	echo "error: invalid option: $OPT $ARG"
	return 1
}

crossbowParseSysConfArg () {
	result=0
	return $result
}

crossbowSetSysArgs () {
	return 0
}
