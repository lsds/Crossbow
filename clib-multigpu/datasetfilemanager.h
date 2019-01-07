#ifndef __CROSSBOW_DATASETFILEMANAGER_H_
#define __CROSSBOW_DATASETFILEMANAGER_H_

#include "memoryregistry.h"
#include "waitfreequeue.h"

#include "memoryregionpool.h"

#include <pthread.h>

typedef struct crossbow_datasetfilemanager *crossbowDatasetFileManagerP;
typedef struct crossbow_datasetfilemanager {
	crossbowMemoryRegistryP registry;
	/* Use of GPU register or not */
	unsigned gpu;
	/* Registration block (i.e. task size) and padding required for page alignment */
	int blocksize;
	int pad;
	/* Use a copy for all files in the registry */
	unsigned copyconstructor;
	/* A pool of temporary memory regions used to copy data set files */
	crossbowMemoryRegionPoolP pool;
} crossbow_datasetfilemanager_t;

crossbowDatasetFileManagerP crossbowDatasetFileManagerCreate (int, unsigned, int);

void crossbowDatasetFileManagerRegister (crossbowDatasetFileManagerP, int, const char *);

void crossbowDatasetFileManagerFree (crossbowDatasetFileManagerP);

#endif /* __CROSSBOW_DATASETFILEMANAGER_H_ */
