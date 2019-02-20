#include "timer.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowTimerP crossbowTimerCreate () {
	crossbowTimerP t = (crossbowTimerP) crossbowMalloc (sizeof(crossbow_timer_t));
	crossbowTimerClear(t);
	return t;
}

void crossbowTimerStart (crossbowTimerP t) {
	gettimeofday(&(t->start), NULL);
	t->isRunning = 1;
	return;
}

int crossbowTimerRunning (crossbowTimerP t) {
	return t->isRunning;
}

void crossbowTimerStop (crossbowTimerP t) {
	gettimeofday(&(t->end), NULL);
	t->isRunning = 0;
	return;
}

void crossbowTimerClear (crossbowTimerP t) {
	memset(t, 0, sizeof(crossbow_timer_t));
	return;
}

tstamp_t crossbowTimerElapsedTime (crossbowTimerP t) {
	tstamp_t t_, _t;
	if (t->isRunning)
		crossbowTimerStop(t);
	 t_ = (tstamp_t) (t->start.tv_sec * 1000000L + t->start.tv_usec);
	_t  = (tstamp_t) (  t->end.tv_sec * 1000000L +   t->end.tv_usec);
	return (_t - t_);
}

tstamp_t crossbowTimerLap (crossbowTimerP t) {
	tstamp_t dt = crossbowTimerElapsedTime (t);
	/* Timer has stopped and t->end contains current time */
	t->start.tv_sec  = t->end.tv_sec;
	t->start.tv_usec = t->end.tv_usec;
	t->isRunning = 1;
	return dt;
}

void crossbowTimerFree (crossbowTimerP t) {
	crossbowFree(t, sizeof(crossbow_timer_t));
	return;
}
