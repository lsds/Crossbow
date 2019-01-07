#ifndef __CROSSBOW_MEMORYREGION_H_
#define __CROSSBOW_MEMORYREGION_H_

#include "datasetfile.h"

/*
 * A page-aligned, page-locked memory region
 */
typedef struct crossbow_memory_region *crossbowMemoryRegionP;
typedef struct crossbow_memory_region {
	int id;
    int capacity;
	int limit;
	void *data;
	crossbowMemoryRegionP next;
	crossbowDatasetFileP file;
	volatile unsigned locked;
	volatile unsigned needed;
	void *pool;
} crossbow_memory_region_t;

crossbowMemoryRegionP crossbowMemoryRegionCreate (int, int, void *);

void crossbowMemoryRegionReset (crossbowMemoryRegionP);

void crossbowMemoryRegionSetLimit (crossbowMemoryRegionP, int);

void crossbowMemoryRegionCopyBlock (crossbowMemoryRegionP, int, int, int);

unsigned long crossbowMemoryRegionAddress (crossbowMemoryRegionP);

void crossbowMemoryRegionRegister (crossbowMemoryRegionP, int, int);

void crossbowMemoryRegionAdviceWillNeed (crossbowMemoryRegionP, int, int);

void crossbowMemoryRegionUnregister (crossbowMemoryRegionP, int, int);

void crossbowMemoryRegionAdviceDontNeed (crossbowMemoryRegionP, int, int);

void crossbowMemoryRegionFree (crossbowMemoryRegionP);

#endif /* __CROSSBOW_MEMORYREGION_H_ */
