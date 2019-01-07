#include "operatordependency.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowOperatorDependencyP crossbowOperatorDependencyCreate (crossbowOperatorDependency_t type, crossbowOperatorP guard, unsigned internal) {
	crossbowOperatorDependencyP p;
	p = (crossbowOperatorDependencyP) crossbowMalloc (sizeof(crossbow_operator_dependency_t));
	p->type = type;
	p->guard = guard;
	p->internal = internal;
	return p;
}

void crossbowOperatorDependencyFree (crossbowOperatorDependencyP p) {
	if (! p)
		return;
	crossbowFree (p, sizeof(crossbow_operator_dependency_t));
}
