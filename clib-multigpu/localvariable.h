#ifndef __CROSSBOW_LOCALVARIABLE_H_
#define __CROSSBOW_LOCALVARIABLE_H_

#include "variable.h"
#include "utils.h"
#include "thetaqueue.h"
#include "arraylist.h"

typedef struct crossbow_localvariable *crossbowLocalVariableP;
typedef struct crossbow_localvariable {
	char *name;
	crossbowVariableP theVariable;
	crossbowArrayListP variables; /* Array of crossbowVariableP, one per device */
	crossbowLocalVariable_t type;
	crossbowArrayListP pool; /* Array of crossbowThetaQueueP, one per device */
} crossbow_localvariable_t;

crossbowLocalVariableP crossbowLocalVariableCreate (const char *, crossbowVariableP, int, int, crossbowArrayListP);

crossbowDataBufferP crossbowLocalVariableGetDataBuffer (crossbowLocalVariableP, int, int, int *, int *);

void crossbowLocalVariableResizePool (crossbowLocalVariableP, crossbowArrayListP);

int crossbowLocalVariableReadOnly (crossbowLocalVariableP);

char *crossbowLocalVariableString (crossbowLocalVariableP);

void crossbowLocalVariableFree (crossbowLocalVariableP);

#endif /* __CROSSBOW_LOCALVARIABLESTORE_H_ */
