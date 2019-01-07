#ifndef __CROSSBOW_TASKHANDLER_H_
#define __CROSSBOW_TASKHANDLER_H_

#include "list.h"
#include "waitfreequeue.h"
#include "stream.h"

#include <pthread.h>

typedef struct crossbow_taskhandler *crossbowTaskHandlerP;
typedef struct crossbow_taskhandler {

	volatile int exit;
	int exited;
	unsigned long id;

	pthread_barrier_t barrier;

	crossbowWaitFreeQueueP tasks;

	crossbowArrayListP callbackhandlers;

	/* Core to pin handler thread. */
	int socket;
	int core;
    
	pthread_t thread;

} crossbow_taskhandler_t;

crossbowTaskHandlerP crossbowTaskHandlerCreate (crossbowArrayListP, int, int);

void crossbowTaskHandlerPublish (crossbowTaskHandlerP, crossbowStreamP);

void crossbowTaskHandlerFree (crossbowTaskHandlerP);

#endif /* __CROSSBOW_TASKHANDLER_H_ */
