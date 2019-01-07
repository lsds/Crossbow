#ifndef __CROSSBOW_CALLBACKHANDLER_H_
#define __CROSSBOW_CALLBACKHANDLER_H_

#include "utils.h"

#include "measurementlist.h"

#include "list.h"
#include "waitfreequeue.h"
#include "modelmanager.h"
#include "resulthandler.h"
#include "stream.h"

#include <pthread.h>

typedef struct crossbow_callbackhandler *crossbowCallbackHandlerP;
typedef struct crossbow_callbackhandler {

	volatile int exit;
	int exited;
	unsigned long id;

	pthread_barrier_t barrier;

	crossbowWaitFreeQueueP events;

	crossbowModelManagerP modelmanager;
	crossbowResultHandlerP resulthandler;
	crossbowArrayListP streams;

	/* Core to pin handler thread */
	int core;

	pthread_t thread;

#ifdef INTRA_TASK_MEASUREMENTS
	crossbowMeasurementListP measurements;
#endif

} crossbow_callbackhandler_t;

crossbowCallbackHandlerP crossbowCallbackHandlerCreate (crossbowModelManagerP, crossbowResultHandlerP, crossbowArrayListP, int);

void crossbowCallbackHandlerPublish (crossbowCallbackHandlerP, crossbowStreamP);

void crossbowCallbackHandlerFree (crossbowCallbackHandlerP);

#endif /* __CROSSBOW_CALLBACKHANDLER_H_ */
