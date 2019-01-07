#include "operator.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "operatordependency.h"

crossbowOperatorP crossbowOperatorCreate (crossbowKernelP kernel, int id) {
	crossbowOperatorP p;
	p = (crossbowOperatorP) crossbowMalloc (sizeof(crossbow_operator_t));
	p->id = id;
	p->kernel = kernel;
	p->peer = NULL;
	p->upstream = NULL;
	p->downstream = NULL;
	p->prev = NULL;
	p->next = NULL;
	p->provider = NULL;
	p->position = 0;
	/* The default sub-stream is 0 */
	p->branch = 0;
	p->events = 0;
	p->start  = NULL;
	p->end    = NULL;
	p->required = 0;
	p->deps = crossbowListCreate ();
	return p;
}

void crossbowOperatorConfigure (crossbowOperatorP p, int events) {
	int i;
	nullPointerException (p);
	p->events = events;
	p->start = crossbowMalloc (p->events * sizeof(cudaEvent_t));
	p->end   = crossbowMalloc (p->events * sizeof(cudaEvent_t));
	for (i = 0; i < p->events; ++i) {
		// checkCudaErrors(cudaEventCreateWithFlags(&(p->start[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
		// checkCudaErrors(cudaEventCreateWithFlags(&(p->end  [i]),  cudaEventBlockingSync | cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&(p->start[i]),  cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&(p->end  [i]),  cudaEventDisableTiming));
	}
	return;
}

int crossbowOperatorGetOutputBufferFromElsewhere (crossbowOperatorP p) {
	return (p->provider != NULL);
}

void crossbowOperatorSetTaskDependency (crossbowOperatorP p, crossbowOperatorDependency_t type, crossbowOperatorP q, unsigned internal) {
	nullPointerException (p);
	crossbowListAppend(p->deps, crossbowOperatorDependencyCreate (type, q, internal));
	return;
}

void crossbowOperatorSchedule (crossbowOperatorP p, void *args) {
	/* Call function pointer */
	p->kernel->func (args);
}

unsigned crossbowOperatorIsMostDownstream (crossbowOperatorP p) {
	return (p->downstream == NULL);
}

void crossbowOperatorFree (crossbowOperatorP p) {
	int i;
	crossbowOperatorDependencyP dependency;
	
	if (! p)
		return;
	
	if (p->upstream)
		crossbowArrayListFree (p->upstream);
	if (p->downstream)
		crossbowArrayListFree (p->downstream);
	
	/* Free events */
	if (p->events > 0) {
		for (i = 0; i < p->events; ++i) {
			checkCudaErrors(cudaEventDestroy(p->start[i]));
			checkCudaErrors(cudaEventDestroy(p->end  [i]));
		}
		crossbowFree (p->start, (p->events * sizeof(cudaEvent_t)));
		crossbowFree (p->end,   (p->events * sizeof(cudaEvent_t)));
	}
	/* Free dependencies array */
	while (! crossbowListEmpty (p->deps)) {
		dependency = (crossbowOperatorDependencyP) crossbowListRemoveFirst (p->deps);
		crossbowOperatorDependencyFree (dependency);
	}
	crossbowListFree (p->deps);

	crossbowFree (p, sizeof(crossbow_operator_t));
	return;
}
