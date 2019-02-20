#ifndef __CROSSBOW_BATCH_H_
#define __CROSSBOW_BATCH_H_

#include "variableschema.h"

typedef struct crossbow_batch *crossbowBatchP;
typedef struct crossbow_batch {

	/* TODO Add documentation about splits */
    int splits;

    crossbowVariableSchemaP examples;
	crossbowVariableSchemaP labels;
} crossbow_batch_t;

crossbowBatchP crossbowBatchCreate ();

void crossbowBatchSetExampleSchema (crossbowBatchP, crossbowVariableSchemaP);

crossbowVariableSchemaP crossbowBatchGetExampleSchema (crossbowBatchP);

void crossbowBatchSetLabelSchema (crossbowBatchP, crossbowVariableSchemaP);

crossbowVariableSchemaP crossbowBatchGetLabelSchema (crossbowBatchP);

int crossbowBatchConfigured (crossbowBatchP);

void crossbowBatchSetSplits(crossbowBatchP, int);

int crossbowBatchGetSplits (crossbowBatchP);

void crossbowBatchFree (crossbowBatchP);

#endif /* __CROSSBOW_BATCH_H_ */
