#ifndef __CROSSBOW_LIGHTWEIGHTDATASETPROCESSOR_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETPROCESSOR_H_

#include "list.h"

#include "lightweightdatasetprocessortask.h"

#include "lightweightdatasetprocessortaskpool.h"

#include <pthread.h>

typedef struct crossbow_lightweightdatasetprocessor *crossbowLightWeightDatasetProcessorP;
typedef struct crossbow_lightweightdatasetprocessor {

	volatile int exit;

	int exited;

	unsigned long id;

	pthread_mutex_t lock;
	pthread_cond_t cond;

	pthread_mutex_t sync; /* Mutex to protect freeList */
	crossbowLightWeightDatasetProcessorTaskP freeList;
	crossbowLightWeightDatasetProcessorTaskPoolP pool;

	crossbowListP events;

	int offset;

	pthread_t thread;

} crossbow_lightweightdatasetprocessor_t;

crossbowLightWeightDatasetProcessorP crossbowLightWeightDatasetProcessorCreate (int);

crossbowLightWeightDatasetProcessorTaskP crossbowLightWeightDatasetProcessorGetTask (crossbowLightWeightDatasetProcessorP);

void crossbowLightWeightDatasetProcessorPutTask (crossbowLightWeightDatasetProcessorP, crossbowLightWeightDatasetProcessorTaskP);

void crossbowLightWeightDatasetProcessorPutTaskSafely (crossbowLightWeightDatasetProcessorP, crossbowLightWeightDatasetProcessorTaskP);

crossbowLightWeightDatasetProcessorTaskP crossbowLightWeightDatasetProcessorCreateTaskPool (crossbowLightWeightDatasetProcessorP, int);

void crossbowLightWeightDatasetProcessorPublish (crossbowLightWeightDatasetProcessorP, crossbowLightWeightDatasetProcessorTaskP);

void crossbowLightWeightDatasetProcessorFree (crossbowLightWeightDatasetProcessorP);

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETPROCESSOR_H_ */
