#ifndef __CROSSBOW_DATASETFILEBLOCK_H_
#define __CROSSBOW_DATASETFILEBLOCK_H_

#include "datasetfile.h"

typedef struct crossbow_datasetfile_block *crossbowDatasetFileBlockP;
typedef struct crossbow_datasetfile_block {
	crossbowDatasetFileBlockP next;
	crossbowDatasetFileP file;
	int offset;
	int length;
	int pad;
	unsigned gpu;
	/* 1 for register, 0 for unregister */
	unsigned op;
} crossbow_datasetfile_block_t;

#endif /* __CROSSBOW_DATASETFILEBLOCK_H_ */
