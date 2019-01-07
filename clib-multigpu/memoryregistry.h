#ifndef __CROSSBOW_MEMORYREGISTRY_H_
#define __CROSSBOW_MEMORYREGISTRY_H_

#include "memoryregistrynode.h"

/*
 * A collection of mapped, page-locked memory regions
 */
typedef struct crossbow_memory_registry *crossbowMemoryRegistryP;
typedef struct crossbow_memory_registry {
	int size;
	crossbowMemoryRegistryNodeP head;
	crossbowMemoryRegistryNodeP tail;
	crossbowMemoryRegistryNodeP list;
} crossbow_memory_registry_t;

crossbowMemoryRegistryP crossbowMemoryRegistryCreate (int);

int crossbowMemoryRegistrySize (crossbowMemoryRegistryP);

crossbowMemoryRegistryNodeP crossbowMemoryRegistryGet (crossbowMemoryRegistryP, int);

void crossbowMemoryRegistryFree (crossbowMemoryRegistryP, int);

#endif /* __CROSSBOW_MEMORYREGISTRY_H_ */
