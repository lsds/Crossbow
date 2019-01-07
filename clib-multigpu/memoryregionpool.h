#ifndef __CROSSBOW_MEMORYREGIONPOOL_H_
#define __CROSSBOW_MEMORYREGIONPOOL_H_

#include "memoryregion.h"

#include <pthread.h>

typedef struct crossbow_memory_region_pool *crossbowMemoryRegionPoolP;
typedef struct crossbow_memory_region_pool {
	int size;
    int autoincrement;
	int capacity;
	pthread_mutex_t sync; /* Mutex to protect list */
	crossbowMemoryRegionP list;
} crossbow_memory_region_pool_t;

crossbowMemoryRegionPoolP crossbowMemoryRegionPoolCreate (int);

crossbowMemoryRegionP crossbowMemoryRegionPoolGet (crossbowMemoryRegionPoolP);

void crossbowMemoryRegionPoolRelease (crossbowMemoryRegionPoolP, crossbowMemoryRegionP);

void crossbowMemoryRegionPoolFree (crossbowMemoryRegionPoolP);

#endif /* __CROSSBOW_MEMORYREGIONPOOL_H_ */
