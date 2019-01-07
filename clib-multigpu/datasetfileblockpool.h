#ifndef __CROSSBOW_DATASETFILEBLOCKPOOL_H_
#define __CROSSBOW_DATASETFILEBLOCKPOOL_H_

#include "datasetfileblock.h"

typedef struct crossbow_datasetfile_block_pool *crossbowDatasetFileBlockPoolP;
typedef struct crossbow_datasetfile_block_pool {
	crossbowDatasetFileBlockP block;
	int size;
	crossbowDatasetFileBlockPoolP next;
} crossbow_datasetfile_block_pool_t;

#endif /* __CROSSBOW_DATASETFILEBLOCKPOOL_H_ */
