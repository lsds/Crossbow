#ifndef __CROSSBOW_KERNELVARIABLE_H_
#define __CROSSBOW_KERNELVARIABLE_H_

#include "variableschema.h"
#include "databuffer.h"

typedef struct crossbow_variable *crossbowVariableP;
typedef struct crossbow_variable {
	crossbowVariableSchemaP schema;
#ifndef __INPUT_ISPINNED_
	void *data;
#endif
	crossbowDataBufferP buffer;
	int offset;

	/* Each model variable might have a different
	 * weight when learning. */
	float learningRateMultiplier;
    
    /* */
    unsigned shifted;
    int offset_;

	crossbowVariableP next;
} crossbow_variable_t;

crossbowVariableP crossbowVariableCreate (crossbowVariableSchemaP);

crossbowDataBufferP crossbowVariableGetDataBuffer (crossbowVariableP, int *, int *);

void crossbowVariableSetLearningRateMultiplier (crossbowVariableP, float);

void crossbowVariableSetDataBuffer (crossbowVariableP, crossbowDataBufferP, int);

void crossbowVariableShift (crossbowVariableP, int);

void crossbowVariableReset (crossbowVariableP);

void crossbowVariableSetHostData (crossbowVariableP, void *, int, int);

void crossbowVariableUnregisterHostData (crossbowVariableP p);

void crossbowVariablePush (crossbowVariableP, cudaStream_t);

crossbowVariableP crossbowVariableReplicate (crossbowVariableP, crossbowDataBufferP);

char *crossbowVariableString (crossbowVariableP);

void crossbowVariableFree (crossbowVariableP);

#endif /* __CROSSBOW_KERNEL_H_ */
