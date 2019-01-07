#include "taskhandler.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "stream.h"

#include "callbackhandler.h"

#include "operatordependency.h"

#include <pthread.h>

#include <unistd.h>
#include <sched.h>

#include <cuda.h>
#include <cuda_runtime.h>

static unsigned long autoincrement = 0UL;

static void crossbowStreamExecute (crossbowTaskHandlerP taskhandler, crossbowStreamP s) {

	int i;
	crossbowOperatorDependencyP d;

	/* Redirect all CUDA calls to specific device */
	checkCudaErrors (cudaSetDevice(s->deviceId));

	dbg("Current stream is %d\n", s->id);

	int examples = crossbowVariableSchemaCountElementsFrom (s->examples->schema, 0) * 4;
	int labels   = crossbowVariableSchemaCountElementsFrom (s->labels->schema,   0) * 4;

	/* Double-check that there are no pending asynchronous tasks on this CUDA streams */
	checkCudaErrors(cudaEventQuery(s->event));

#ifdef INTRA_TASK_MEASUREMENTS
	/* Record start of task */
	checkCudaErrors(cudaEventRecord(s->start, s->stream[0]));
#endif

	/* Input data movement to stream zero */
#ifdef __INPUT_ISPINNED_
	/* Push input variables in one go, since the input data have been copied in continuous memory regions */
	crossbowDataBufferPush (s->input, s->stream[0]);
#else
	/* Input variables have to be copied one-by-one */
	crossbowVariablePush (s->examples, s->stream[0]);
	crossbowVariablePush (s->labels,   s->stream[0]);
#endif

	/* Set cuRAND stream.
	 *
	 * cuRAND is used by the data transformation operator, so we set the stream accordingly.
	 * Nonetheless, we don't record dependencies for data movement.
	 */
	if (s->dataflow->dataTransformOp)
		checkCurandStatus(curandSetStream(s->curandGenerator, s->stream[s->dataflow->dataTransformOp->branch]));

	/* The cuBLAS and cuDNN streams are set appropriately at creation time. */

	for (i = 0; i < s->splits; ++i) {

		/* Iterate over dataflow operators and schedule kernels */
		s->op = crossbowDataflowPeek (s->dataflow);

		/*
		 * Compute input checksum (for debugging purposes)
		 * crossbowStreamComputeInputCheckSum (s);
		 */

		while (s->op != s->dataflow->tail) {
			dbg("Schedule kernel function(s) for op %s\n", s->op->kernel->name);

			/* Intra-task dependencies */
			if (s->branches > 1 && (! crossbowListEmpty (s->op->deps))) {
				crossbowListIteratorReset (s->op->deps);
				while (crossbowListIteratorHasNext(s->op->deps)) {
					d = (crossbowOperatorDependencyP) crossbowListIteratorNext (s->op->deps);
					checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], d->guard->end[s->id], 0));
				}
			}

			crossbowOperatorSchedule (s->op, s); /* Schedule current operator's kernel */

			if (s->branches > 1) {
				if (s->op->required > 0 || crossbowOperatorIsMostDownstream (s->op))
					checkCudaErrors(cudaEventRecord(s->op->end[s->id], s->stream[s->op->branch]));
			}

			#ifdef UPDATE_MODEL_INCREMENTALLY
			crossbowStreamUpdateModel (s);
			#endif

			#ifdef COMPUTE_CHECKSUM
			crossbowStreamComputeCheckSum (s); /* Compute current operator's checksum  */
			#endif

			s->op = s->op->next; /* Get next operator */
		}

		/* Finished with one split:
		 *
		 * a) Move input data buffer by the (actual) batch size;
		 * b) Move labels variable offset by the (actual) batch size.
		 */
		if (i < (s->splits - 1)) {
			crossbowDataBufferShift (s->input, examples);
			crossbowVariableShift   (s->labels, (labels - examples));
		}
	}

	/*
	 * Making sure that all operators have finished before scheduling any output data movement.
	 * The assumption here is that we can wait for all downstream operators to finish.
	 */
	if (s->branches > 1) {
		s->op = crossbowDataflowPeek (s->dataflow);
		while (s->op != s->dataflow->tail) {
			if (crossbowOperatorIsMostDownstream (s->op))
				checkCudaErrors(cudaStreamWaitEvent(s->stream[0], s->op->end[s->id], 0));
			s->op = s->op->next; /* Get next operator */
		}
	}

	/* Perform asynchrounous SGD here? */

	/* Iterate once again, this time scheduling any output data movement */
	s->op = crossbowDataflowPeek (s->dataflow);
	while (s->op != s->dataflow->tail) {
		if (crossbowKernelOutputPull(s->op->kernel)) {
			dbg("Schedule output data movement operation(s) for op %s\n", s->op->kernel->name);
			crossbowDataBufferPull(crossbowListPeekHead(s->outputs[s->op->id]), s->stream[0]);
		}
		s->op = s->op->next;
	}
    
	/* Record task completion event */
	checkCudaErrors(cudaEventRecord(s->event, s->stream[0]));

	/* Handle task completion event */
    crossbowArrayListP pool = (crossbowArrayListP) crossbowArrayListGet (taskhandler->callbackhandlers, taskhandler->socket);
	crossbowCallbackHandlerP callbackhandler = crossbowArrayListGetNextSafely (pool);
	dbg("Task handler #%lu assigns task %04d to callback handler #%lu\n", taskhandler->id, s->task, callbackhandler->id);
	crossbowCallbackHandlerPublish (callbackhandler, s);

	return;
}

static void *handle (void *args) {

	crossbowTaskHandlerP self = NULL;
	crossbowStreamP s = NULL;

	self = (crossbowTaskHandlerP) args;
	
	if (self->core >= 0) {
		cpu_set_t set;
		CPU_ZERO (&set);
		int core = self->core; // self->socket + self->id;
		CPU_SET  (core, &set);
		sched_setaffinity (0, sizeof(set), &set);
		info("Task handler #%02lu pinned on core %2d\n", self->id, core);
	}

	dbg("Task handler #%lu starts\n", self->id);
	pthread_barrier_wait (&(self->barrier));

	while (! self->exit) {
		/* Busy-wait for a task */
		s = (crossbowStreamP) crossbowWaitFreeQueueDequeueOrWait (self->tasks);
		if (! s)
			break;
		crossbowStreamExecute (self, s);
	}
	self->exited = 1;
	return self;
}

crossbowTaskHandlerP crossbowTaskHandlerCreate (crossbowArrayListP callbackhandlers, int socket, int core) {
	crossbowTaskHandlerP p;
	p = (crossbowTaskHandlerP) crossbowMalloc (sizeof(crossbow_taskhandler_t));
	p->exit = 0;
	p->exited = 0;
	p->id = autoincrement++;
	/* Create a barrier between the main thread and a task handler */
	pthread_barrier_init(&(p->barrier), NULL, 2);
	p->tasks = crossbowWaitFreeQueueCreate(10);
	p->callbackhandlers = callbackhandlers;
	p->socket = socket;
	p->core = core;
	/* Launch thread */
	pthread_create (&(p->thread), NULL, handle, (void *) p);
	/* Wait for callback handler to be initialised */
	pthread_barrier_wait (&(p->barrier));
	return p;
}

void crossbowTaskHandlerPublish (crossbowTaskHandlerP p, crossbowStreamP args) {
	/* Should we wait for a slot to be freed? */
	crossbowWaitFreeQueueEnqueueOrWait(p->tasks, args);
}

void crossbowTaskHandlerFree (crossbowTaskHandlerP p) {
	if (! p->exited) {
		p->exit = 1;
		/* Unblock thread */
		crossbowWaitFreeQueueEnqueue(p->tasks, NULL);
	}
	/* Wait until thread has exited */
	pthread_join(p->thread, NULL);
	/* Queue of events should be empty */
	crossbowWaitFreeQueueFree (p->tasks);
	pthread_barrier_destroy (&(p->barrier));
	/* Task queue should be empty */
	crossbowFree (p, sizeof(crossbow_taskhandler_t));
}
