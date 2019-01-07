#ifndef __CROSSBOW_MEMORYREGISTRYNODE_H_
#define __CROSSBOW_MEMORYREGISTRYNODE_H_

#include "datasetfile.h"

typedef struct crossbow_memory_registry_node *crossbowMemoryRegistryNodeP;
typedef struct crossbow_memory_registry_node {
	crossbowMemoryRegistryNodeP next;
	crossbowDatasetFileP file;
} crossbow_memory_registry_node_t;

#endif /* __CROSSBOW_MEMORYREGISTRYNODE_H_ */
