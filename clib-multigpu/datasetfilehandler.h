#ifndef __CROSSBOW_DATASETFILEHANDLER_H_
#define __CROSSBOW_DATASETFILEHANDLER_H_

#include "list.h"

#include "datasetfileblock.h"

#include "datasetfileblockpool.h"

#include <pthread.h>

typedef struct crossbow_datasetfilehandler *crossbowDatasetFileHandlerP;
typedef struct crossbow_datasetfilehandler {

	volatile int exit;

	int exited;

	unsigned long id;

	pthread_mutex_t lock;
	pthread_cond_t cond;

	pthread_mutex_t sync; /* Mutex to protect freeList */
	crossbowDatasetFileBlockP freeList;
	crossbowDatasetFileBlockPoolP pool;

	crossbowListP events;

	int offset;

	pthread_t thread;

} crossbow_datasetfilehandler_t;

crossbowDatasetFileHandlerP crossbowDatasetFileHandlerCreate (int);

crossbowDatasetFileBlockP crossbowDatasetFileHandlerGetBlock (crossbowDatasetFileHandlerP);

crossbowDatasetFileBlockP crossbowDatasetFileHandlerCreateBlockPool (crossbowDatasetFileHandlerP, int);

void crossbowDatasetFileHandlerPutBlock (crossbowDatasetFileHandlerP, crossbowDatasetFileBlockP);

void crossbowDatasetFileHandlerPutBlockSafely (crossbowDatasetFileHandlerP, crossbowDatasetFileBlockP);

void crossbowDatasetFileHandlerPublish (crossbowDatasetFileHandlerP, crossbowDatasetFileBlockP);

void crossbowDatasetFileHandlerFree (crossbowDatasetFileHandlerP);

#endif /* __CROSSBOW_DATASETFILEHANDLER_H_ */
