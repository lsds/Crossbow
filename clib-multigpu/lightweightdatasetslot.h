#ifndef __CROSSBOW_LIGHTWEIGHTDATASETSLOT_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETSLOT_H_

#include "lightweightdatasetbuffer.h"

#include "lightweightdatasettask.h"

typedef struct crossbow_lightweightdatasetslot *crossbowLightWeightDatasetSlotP;
typedef struct crossbow_lightweightdatasetslot {
	int id;

	crossbowLightWeightDatasetTaskP table;
	int ndx; /*         Current task index */
	int inc; /* Offset for next task index */
	int max; /*         Maximum task index */

	crossbowLightWeightDatasetBufferP buffer;
	int offset;
	int length;

	int epochs;

} crossbow_lightweightdatasetslot_t;

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETSLOT_H_ */
