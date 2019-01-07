#include "datasetfilemanager.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "datasetfile.h"

crossbowDatasetFileManagerP crossbowDatasetFileManagerCreate (int size, unsigned gpu, int blocksize) {
	crossbowDatasetFileManagerP p;
	p = (crossbowDatasetFileManagerP) crossbowMalloc(sizeof(crossbow_datasetfilemanager_t));
	p->registry = crossbowMemoryRegistryCreate (size);
	p->gpu = gpu;
	p->blocksize = blocksize;
	p->pad = 0;
	p->copyconstructor = 0;
	p->pool = NULL;
	return p;
}

void crossbowDatasetFileManagerRegister (crossbowDatasetFileManagerP p, int id, const char *filename) {
	crossbowMemoryRegistryNodeP node = crossbowMemoryRegistryGet (p->registry, id);
	nullPointerException(node);
	crossbowDatasetFileP file = crossbowDatasetFileCreate (filename);
	node->file = file;
	return;
}

void crossbowDatasetFileManagerFree (crossbowDatasetFileManagerP p) {
	if (! p)
		return;
	crossbowMemoryRegistryFree (p->registry, p->blocksize);
	if (p->copyconstructor)
        crossbowMemoryRegionPoolFree (p->pool);
	crossbowFree(p, sizeof(crossbow_datasetfilemanager_t));
	return;
}
