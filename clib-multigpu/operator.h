#ifndef __CROSSBOW_OPERATOR_H_
#define __CROSSBOW_OPERATOR_H_

#include "kernel.h"
#include "list.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

typedef struct crossbow_operator *crossbowOperatorP;
typedef struct crossbow_operator {
	int id;
	crossbowKernelP kernel;
	crossbowOperatorP peer; /* If not null, then this operator is the gradient for operator `peer` */
	/* Upstream and downstream operators in dataflow */
	crossbowArrayListP upstream, downstream;
	/* Previous and next operator after topological sort */
	crossbowOperatorP prev, next;
	/* Operator that provides output (according to memory plan), NULL otherwise */
	crossbowOperatorP provider;
	int position;
	/* Sub-stream on which this operator should be scheduled */
	int branch;
	/* Events marking the start and end of this operator on a stream */
	int events;
	cudaEvent_t *start;
	cudaEvent_t *end;
	int required;
	/* Task dependencies */
	crossbowListP deps;
} crossbow_operator_t;

crossbowOperatorP crossbowOperatorCreate (crossbowKernelP, int);

int crossbowOperatorGetOutputBufferFromElsewhere (crossbowOperatorP);

void crossbowOperatorConfigure (crossbowOperatorP, int);

void crossbowOperatorSetTaskDependency (crossbowOperatorP, crossbowOperatorDependency_t, crossbowOperatorP, unsigned);

void crossbowOperatorSchedule (crossbowOperatorP, void *);

unsigned crossbowOperatorIsMostDownstream (crossbowOperatorP);

void crossbowOperatorFree (crossbowOperatorP);

#endif /* __CROSSBOW_OPERATOR_H_ */
