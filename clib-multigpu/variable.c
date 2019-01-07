#include "variable.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <stdint.h>

crossbowVariableP crossbowVariableCreate (crossbowVariableSchemaP schema) {
	crossbowVariableP p = (crossbowVariableP) crossbowMalloc (sizeof(crossbow_variable_t));
	p->schema = schema;
#ifndef __INPUT_ISPINNED_
	p->data = NULL;
#endif
	p->buffer = NULL;
	p->offset = 0;
	p->learningRateMultiplier = 1;
    p->shifted = 0;
    p->offset_ = 0;
	p->next = NULL;
	return p;
}

crossbowDataBufferP crossbowVariableGetDataBuffer (crossbowVariableP p, int *offset, int *length) {
	nullPointerException(p->buffer);
	if (offset)
		*offset = p->offset;
	if (length)
		*length = p->schema->bytes;
	return p->buffer;
}

void crossbowVariableSetLearningRateMultiplier (crossbowVariableP p, float learningRateMultiplier) {
	nullPointerException (p);
	p->learningRateMultiplier = learningRateMultiplier;
	return;
}

void crossbowVariableSetDataBuffer (crossbowVariableP p, crossbowDataBufferP buffer, int offset) {
	p->buffer = buffer;
	p->offset = offset;
	/* Store original offset */
	p->offset_= offset;
	return;
}

void crossbowVariableShift (crossbowVariableP p, int bytes) {
	nullPointerException (p);
	dbg("Shift variable offset by %d bytes (current offset is %d)\n", bytes, p->offset);
	p->offset += bytes;
	invalidArgumentException (p->offset >= 0);
	p->shifted += 1;
}

void crossbowVariableReset (crossbowVariableP p) {
	nullPointerException (p);
	if (p->shifted) {
		p->offset = p->offset_;
		p->shifted = 0;
	}
	return;
}

void crossbowVariableSetHostData (crossbowVariableP p, void *buffer, int start, int end) {
	int length = end - start;
	invalidArgumentException(p->schema->bytes == length);
#ifndef __INPUT_ISPINNED_
	p->data = (void *) ((char *) (buffer) + start);
#else
	crossbowDataBufferInitHostRegion (p->buffer, p->offset, buffer, start, length);
#endif
	return;
}

void crossbowVariablePush (crossbowVariableP p, cudaStream_t stream) {
#ifndef __INPUT_ISPINNED_
	nullPointerException(p->data);
	crossbowDataBufferPushRegion (p->buffer, p->data, p->offset, p->schema->bytes, stream);
#else
	(void) p;
	(void) stream;
	illegalOperationException();
#endif
	return;
}

crossbowVariableP crossbowVariableReplicate (crossbowVariableP p, crossbowDataBufferP buffer) {
	crossbowVariableP copy;
	copy = crossbowVariableCreate (crossbowVariableSchemaCopy (p->schema));
	/* Same pointer but to a different buffer */
	copy->offset = p->offset;
	copy->learningRateMultiplier = p->learningRateMultiplier;
	copy->buffer = buffer;
	return copy;
}

char *crossbowVariableString (crossbowVariableP p) {
	return crossbowVariableSchemaString (p->schema);
}

void crossbowVariableFree (crossbowVariableP p) {
	crossbowVariableSchemaFree (p->schema);
	crossbowFree (p, sizeof(crossbow_variable_t));
	return;
}
