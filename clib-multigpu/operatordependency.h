#ifndef __CROSSBOW_OPERATOR_DEPENDENCY_H_
#define __CROSSBOW_OPERATOR_DEPENDENCY_H_

#include "utils.h"

#include "operator.h"

typedef struct crossbow_operator_dependency *crossbowOperatorDependencyP;
typedef struct crossbow_operator_dependency {
	crossbowOperatorDependency_t type;
	crossbowOperatorP guard;
	unsigned internal;
} crossbow_operator_dependency_t;

crossbowOperatorDependencyP crossbowOperatorDependencyCreate (crossbowOperatorDependency_t, crossbowOperatorP, unsigned);

void crossbowOperatorDependencyFree (crossbowOperatorDependencyP);

#endif /* __CROSSBOW_OPERATOR_DEPENDENCY_H_ */
