#include "variableschema.h"

#include "memorymanager.h"

#include "utils.h"
#include "debug.h"

crossbowVariableSchemaP crossbowVariableSchemaCreate (int dims, int *shape, int bytes) {
	int i;
	crossbowVariableSchemaP p = (crossbowVariableSchemaP) crossbowMalloc (sizeof(crossbow_variable_schema_t));
	p->dims = dims;
	p->shape = (int *) crossbowMalloc (p->dims * sizeof(int));
	p->elements = 1;
	for (i = 0; i < p->dims; i++) {
		p->shape[i] = shape[i];
		p->elements *= p->shape[i];
	}
	p->bytes = bytes;
	return p;
}

crossbowVariableSchemaP crossbowVariableSchemaCopy (crossbowVariableSchemaP p) {
	return crossbowVariableSchemaCreate (p->dims, p->shape, p->bytes);
}

int crossbowVariableSchemaEqual (crossbowVariableSchemaP p, crossbowVariableSchemaP q) {
	/* Returns one if equal, zero otherwise */
	return memcmp (p, q, sizeof(crossbow_variable_schema_t));
}

int crossbowVariableSchemaCountElementsInRange (crossbowVariableSchemaP p, int start, int end) {
	int axis;
	int count = 1;
	for (axis = start; axis < end; ++axis)
		count *= p->shape[axis];
	return count;
}

int crossbowVariableSchemaCountElementsFrom (crossbowVariableSchemaP p, int offset) {
	return crossbowVariableSchemaCountElementsInRange (p, offset, p->dims);
}

int crossbowVariableSchemaShape (crossbowVariableSchemaP p, int ndx) {
	return p->shape[ndx];
}

char *crossbowVariableSchemaString (crossbowVariableSchemaP p) {
	int i;
	char s [1024];
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	/* [a, b, ...] (X elements Y bytes) */
	crossbowStringAppend (s, &offset, &remaining, "[");
	// info("fill in %d dims...\n", p->dims);
	for (i = 0; i < p->dims; i++) {
		crossbowStringAppend (s, &offset, &remaining, "%d", p->shape[i]);
		if (i < (p->dims - 1))
			crossbowStringAppend (s, &offset, &remaining, ", ");
		else
			crossbowStringAppend (s, &offset, &remaining, "] ");
	}
	crossbowStringAppend (s, &offset, &remaining, "(%d element%s %d byte%s)", p->elements, (p->elements == 1) ? "" : "s", p->bytes, (p->bytes == 1) ? "" : "s");
	return crossbowStringCopy (s);
}

void crossbowVariableSchemaFree (crossbowVariableSchemaP p) {
	crossbowFree (p->shape, p->dims * sizeof(int));
	crossbowFree (p, sizeof(crossbow_variable_schema_t));
	return;
}
