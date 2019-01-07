#ifndef __CROSSBOW_LIGHTWEIGHTDATASETMANAGER_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETMANAGER_H_

#include "memoryregistry.h"

#include "lightweightdatasetbuffer.h"

#include "lightweightdatasettask.h"
#include "lightweightdatasetslot.h"

#include <pthread.h>

typedef struct crossbow_lightweightdatasetmanager *crossbowLightWeightDatasetManagerP;
typedef struct crossbow_lightweightdatasetmanager {

	crossbowMemoryRegistryP registry;

	/* Use of GPU register or not */
	unsigned GPU;

	/* Block size (i.e. task size) and padding required for page alignment */
	int blocksize;
	int   padding;

	crossbowLightWeightDatasetBufferP buffer;

	int numberofslots;
	int numberoftasks;

	crossbowLightWeightDatasetSlotP slots;
	crossbowLightWeightDatasetTaskP tasks;

} crossbow_lightweightdatasetmanager_t;

crossbowLightWeightDatasetManagerP crossbowLightWeightDatasetManagerCreate (int, unsigned, int);

void crossbowLightWeightDatasetManagerRegister (crossbowLightWeightDatasetManagerP, int, const char *);

void crossbowLightWeightDatasetManagerCreateSlots (crossbowLightWeightDatasetManagerP, int);

void crossbowLightWeightDatasetManagerCreateTasks (crossbowLightWeightDatasetManagerP, int, int, int);

void crossbowLightWeightDatasetManagerDump (crossbowLightWeightDatasetManagerP);

void crossbowLightWeightDatasetManagerFree (crossbowLightWeightDatasetManagerP);

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETMANAGER_H_ */
