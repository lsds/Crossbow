#ifndef __CROSSBOW_TIMER_H_
#define __CROSSBOW_TIMER_H_

#include <sys/time.h>

typedef unsigned long long tstamp_t;

typedef struct crossbow_timer *crossbowTimerP;
typedef struct crossbow_timer {
	struct timeval start;
	struct timeval end;
	int isRunning;
} crossbow_timer_t;

crossbowTimerP crossbowTimerCreate ();

void crossbowTimerStart (crossbowTimerP);

int crossbowTimerRunning (crossbowTimerP);

void crossbowTimerStop (crossbowTimerP);

void crossbowTimerClear (crossbowTimerP);

tstamp_t crossbowTimerElapsedTime (crossbowTimerP);

tstamp_t crossbowTimerLap (crossbowTimerP);

void crossbowTimerFree (crossbowTimerP);

#endif /* __CROSSBOW_TIMER_H_ */
