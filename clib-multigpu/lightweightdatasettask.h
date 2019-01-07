#ifndef __CROSSBOW_LIGHTWEIGHTDATASETTASK_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETTASK_H_

#include "datasetfile.h"

typedef struct crossbow_lightweightdatasettask *crossbowLightWeightDatasetTaskP;
typedef struct crossbow_lightweightdatasettask {
	crossbowDatasetFileP file[2];
	int offset;
	int length;
} crossbow_lightweightdatasettask_t;

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETTASK_H_ */
