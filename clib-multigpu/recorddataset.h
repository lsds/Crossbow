#ifndef __CROSSBOW_RECORD_DATASET_H_
#define __CROSSBOW_RECORD_DATASET_H_

#include "doublebuffer.h"
#include "image/recordreader.h"

#include <pthread.h>

typedef struct crossbow_record_dataset *crossbowRecordDatasetP;
typedef struct crossbow_record_dataset {

	/* Pointer to images and labels buffer */
	void *images;
	void *labels;

	int count;

	crossbowRecordReaderP reader;
	crossbowDoubleBufferP buffer;

	/* Create worker thread */

	volatile int exit;
	int exited;

	pthread_mutex_t lock;
	pthread_cond_t cond;

	crossbowListP events;

	pthread_t thread;

} crossbow_record_dataset_t;

typedef struct crossbow_record_dataset_event *crossbowRecordDatasetEventP;
typedef struct crossbow_record_dataset_event {
	int idx;
} crossbow_record_dataset_event_t;

crossbowRecordDatasetP crossbowRecordDatasetCreate (int, int *, int, int, int *);

void crossbowRecordDatasetInit (crossbowRecordDatasetP);

void crossbowRecordDatasetInitSafely (crossbowRecordDatasetP);

void crossbowRecordDatasetSwap (crossbowRecordDatasetP);

void crossbowRecordDatasetFree (crossbowRecordDatasetP);

#endif /* __CROSSBOW_RECORD_DATASET_H_ */
