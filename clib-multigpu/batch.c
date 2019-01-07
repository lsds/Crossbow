#include "batch.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowBatchP crossbowBatchCreate () {
	crossbowBatchP batch;
	batch = (crossbowBatchP) crossbowMalloc (sizeof(crossbow_batch_t));
    batch->splits = 1;
	batch->examples = NULL;
	batch->labels = NULL;
	return batch;
}

void crossbowBatchSetSplits(crossbowBatchP batch, int splits) {
	nullPointerException(batch);
    batch->splits = splits;
}

void crossbowBatchSetExampleSchema (crossbowBatchP batch, crossbowVariableSchemaP schema) {
	nullPointerException(batch);
	nullPointerException(schema);
	batch->examples = schema;
	return;
}

void crossbowBatchSetLabelSchema (crossbowBatchP batch, crossbowVariableSchemaP schema) {
	nullPointerException(batch);
	nullPointerException(schema);
	batch->labels = schema;
	return;
}

crossbowVariableSchemaP crossbowBatchGetExampleSchema (crossbowBatchP batch) {
	nullPointerException(batch);
	return batch->examples;
}

crossbowVariableSchemaP crossbowBatchGetLabelSchema (crossbowBatchP batch) {
	nullPointerException(batch);
	return batch->labels;
}

int crossbowBatchGetSplits (crossbowBatchP batch) {
	nullPointerException(batch);
    return batch->splits;
}

int crossbowBatchConfigured (crossbowBatchP batch) {
	nullPointerException(batch);
	return (batch->examples != NULL && batch->labels != NULL);
}

void crossbowBatchFree (crossbowBatchP batch) {
	if (batch) {
		if (batch->examples)
			crossbowVariableSchemaFree (batch->examples);
		if (batch->labels)
			crossbowVariableSchemaFree (batch->labels);
		crossbowFree (batch, sizeof(crossbow_batch_t));
	}
	return;
}
