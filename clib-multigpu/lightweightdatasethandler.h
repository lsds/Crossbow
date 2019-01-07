#ifndef __CROSSBOW_LIGHTWEIGHTDATASETHANDLER_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETHANDLER_H_

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

typedef struct crossbow_lightweightdatasethandler *crossbowLightWeightDatasetHandlerP;
typedef struct crossbow_lightweightdatasethandler {
	void **slots;
	int *count;
	int offset;
} crossbow_lightweightdatasethandler_t;

crossbowLightWeightDatasetHandlerP crossbowLightWeightDatasetHandlerCreate (int);

void  crossbowLightWeightDatasetHandlerSet (crossbowLightWeightDatasetHandlerP, int, void *, int);

void *crossbowLightWeightDatasetHandlerGet (crossbowLightWeightDatasetHandlerP, int);

int crossbowLightWeightDatasetHandlerNumberOfSlots (crossbowLightWeightDatasetHandlerP, int);

void crossbowLightWeightDatasetHandlerCompareAndSwap (crossbowLightWeightDatasetHandlerP, int, int, int, int);

void crossbowLightWeightDatasetHandlerReady (crossbowLightWeightDatasetHandlerP, int, int);

void crossbowLightWeightDatasetHandlerPrepareToReserve (crossbowLightWeightDatasetHandlerP, int, int);

void crossbowLightWeightDatasetHandlerReserve (crossbowLightWeightDatasetHandlerP, int, int);

void crossbowLightWeightDatasetHandlerPrepareToRelease (crossbowLightWeightDatasetHandlerP, int, int);

void crossbowLightWeightDatasetHandlerRelease (crossbowLightWeightDatasetHandlerP, int, int);

void crossbowLightWeightDatasetHandlerFree (crossbowLightWeightDatasetHandlerP);

static inline int crossbowLightWeightDatasetHandlerTranslate (long ptr, int size) {
	invalidConditionException (size > 0);
	invalidConditionException (((ptr + 1) % size) == 0);
	return (((int) (ptr + 1)) / size) - 1;
}

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETHANDLER_H_ */
