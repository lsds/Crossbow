#ifndef __CROSSBOW_BATCH_H_
#define __CROSSBOW_BATCH_H_

#include "variableschema.h"

typedef struct crossbow_batch *crossbowBatchP;
typedef struct crossbow_batch {
    int splits;
	crossbowVariableSchemaP examples;
	crossbowVariableSchemaP labels;
} crossbow_batch_t;

crossbowBatchP crossbowBatchCreate ();

void crossbowBatchSetSplits(crossbowBatchP, int);

void crossbowBatchSetExampleSchema (crossbowBatchP, crossbowVariableSchemaP);

void crossbowBatchSetLabelSchema (crossbowBatchP, crossbowVariableSchemaP);

crossbowVariableSchemaP crossbowBatchGetExampleSchema (crossbowBatchP);

crossbowVariableSchemaP crossbowBatchGetLabelSchema (crossbowBatchP);

int crossbowBatchGetSplits (crossbowBatchP);

int crossbowBatchConfigured (crossbowBatchP);

void crossbowBatchFree (crossbowBatchP);

#endif /* __CROSSBOW_BATCH_H_ */
