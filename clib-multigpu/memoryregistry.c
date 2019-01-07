#include "memoryregistry.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowMemoryRegistryP crossbowMemoryRegistryCreate (int size) {
	int order;
	crossbowMemoryRegistryP registry;
	crossbowMemoryRegistryNodeP node;
	registry = (crossbowMemoryRegistryP) crossbowMalloc (sizeof(crossbow_memory_registry_t));
	registry->size = size;
	registry->head = NULL;
	registry->tail = NULL;
	registry->list = (crossbowMemoryRegistryNodeP) crossbowMalloc ((registry->size) * sizeof(crossbow_memory_registry_node_t));
	node = registry->list;
	for (order = 0; order < registry->size; order++, node++) {
		node->next = NULL;
		node->file = NULL;
		if (order == 0) /* Set head to be the first element */
			registry->head = node;
		else
			registry->tail->next = node;
		registry->tail = node;
	}
	return registry;
}

int crossbowMemoryRegistrySize (crossbowMemoryRegistryP registry) {
	return registry->size;
}

crossbowMemoryRegistryNodeP crossbowMemoryRegistryGet (crossbowMemoryRegistryP registry, int id) {
	indexOutOfBoundsException(id, registry->size);
	return &(registry->list[id]);
}

void crossbowMemoryRegistryFree (crossbowMemoryRegistryP registry, int block) {
	int count;
	crossbowMemoryRegistryNodeP node;
	if (! registry)
		return;
	node = registry->head;
	count = 0;
	while (node != NULL) {
		crossbowDatasetFileFree (node->file, block);
		count ++;
		node = node->next;
	}
	invalidConditionException (count == registry->size);
	crossbowFree (registry->list, (registry->size) * sizeof(crossbow_memory_registry_node_t));
	crossbowFree (registry, sizeof(crossbow_memory_registry_t));
}
