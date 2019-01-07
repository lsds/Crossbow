#ifndef __CROSSBOW_LIGHTWEIGHTDATASETPROCESSORTASK_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETPROCESSORTASK_H_

#include "lightweightdatasetslot.h"

#include "lightweightdatasethandler.h"

#include "utils.h"

typedef struct crossbow_lightweightdatasetprocessortask *crossbowLightWeightDatasetProcessorTaskP;
typedef struct crossbow_lightweightdatasetprocessortask {
	crossbowLightWeightDatasetProcessorTaskP next;
	crossbowLightWeightDatasetOp_t op;
	unsigned GPU;
	unsigned phi;
	crossbowLightWeightDatasetSlotP slot[2];
	crossbowLightWeightDatasetHandlerP handler;

} crossbow_lightweightdatasetprocessortask_t;

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETPROCESSORTASK_H_ */
