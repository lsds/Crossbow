#include "callbackhandler.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "recorddataset.h"

#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>

static unsigned long autoincrement = 0UL;

static float __get_loss_value (crossbowStreamP s) { /* Get loss value */
	/* Default value is 0 */
	float loss = 0;

	int ndx;
	crossbowDataBufferP b;

	if (! s->dataflow->lossOp)
		return loss;

	ndx = s->dataflow->lossOp->id;
	b = crossbowListPeekHead(s->outputs[ndx]);

	loss = *(float *)(b->host);
	return loss;
}

static float __get_accuracy_value (crossbowStreamP s) { /* Get accuracy value, if phase is CHECK */
	/* Default value is 0 */
	float accuracy = 0;

	int ndx;
	crossbowDataBufferP b;
	/*
	if (s->phi != CHECK)
		return accuracy;
	*/
	if (! s->dataflow->accuracyOp)
		return accuracy;

	ndx = s->dataflow->accuracyOp->id;
	b = crossbowListPeekHead(s->outputs[ndx]);
	accuracy = ((float *) b->host)[0];

	return accuracy;
}

static void *handle (void *args) {

	crossbowCallbackHandlerP self = NULL;
	crossbowStreamP s = NULL;
	int i;
	crossbowDataBufferP output, local;

#ifdef INTRA_TASK_MEASUREMENTS
	float dt;
#endif

	self = (crossbowCallbackHandlerP) args;

	/* Pin handler to core */
	if (self->core > 0) {
		cpu_set_t set;
		int core = self->core; 
		CPU_ZERO (&set);
		CPU_SET  (core, &set);
		sched_setaffinity (0, sizeof(set), &set);
		info("Callback handler #%lu pinned on core %d\n", self->id, core);
	}

	dbg("Callback handler #%lu starts\n", self->id);
	pthread_barrier_wait (&(self->barrier));

	while (! self->exit) {
		/* Busy wait for an event */
		s = (crossbowStreamP) crossbowWaitFreeQueueDequeueOrWait (self->events);
        if (! s)
            break;
        
		/* Synchronise on task's event (busy waiting) */
		checkCudaErrors(cudaEventSynchronize(s->event));
		/*
		 * Note that the alternative is to busy-wait explicitly:
		 *
		 * while (cudaEventQuery(s->event) != cudaSuccess);
		 */
		
		/* 
		 * Notify record dataset that a task completed 
		 * (and therefore its data can be overwritten)
		 */
		if (s->dataset)
			crossbowRecordDatasetNotify (s->dataset);

#ifdef INTRA_TASK_MEASUREMENTS
		checkCudaErrors(cudaEventElapsedTime (&dt, s->start, s->event));
		crossbowMeasurementListAppend (self->measurements, dt);
#endif

#ifdef MAKESPAN_MEASUREMENTS
		checkCudaErrors(cudaEventElapsedTime (&dt, s->barrier, s->event));
		info("Makespan of task %2d is %7.2f us\n", s->task, dt);
#endif
		/* Debugging */
		float loss = __get_loss_value (s);
		dbg("Task %2d loss is %5.5f\n", s->task, loss);
		
        float accuracy = __get_accuracy_value(s);

		/* Release output buffers */
		for (i = 0; i < s->ops; i++) {
			while (! crossbowListEmpty(s->outputs[i])) {
				output = (crossbowDataBufferP) crossbowListRemoveFirst (s->outputs[i]);
				if (--(output->refs) == 0) {
					crossbowDataBufferRelease (output);
				}
			}
		}
		dbg("Released output buffers\n");

		/* Release local variables */
		for (i = 0; i < s->ops; i++) {
			while (! crossbowListEmpty(s->locals[i])) {
				local = crossbowListRemoveFirst(s->locals[i]);
                if (--(local->refs) == 0) {
                	crossbowDataBufferRelease (local);
                }
			}
		}
		dbg("Released local variables\n");

		/* Release model replica
		 *
		 * Note that the model manager is initialised by the application
		 * after the handler started. At start time, it is NULL.
		 *
		 * The model should be released (and unlocked) before the result
		 * handler is invoked because it cause the following error:
		 *
		 * A BSP barrier is reached and the result collector attempts to
		 * acquire all locks but fails
		 */
		nullPointerException(self->modelmanager);

		/* info("Finished task on device %d stream %d with model %d\n", s->deviceId, s->id, s->model->id); */

		/* Increment model updates */
		s->model->updates += 1;
		crossbowModelManagerRelease (self->modelmanager, s->model); /* Model is unlocked */
		dbg("Released model %d\n", s->model->id);

		/* Handle results */
		crossbowResultHandlerReserveSlot (self->resulthandler, s->phi, s->task, &s->freeP[0], loss, accuracy);

		crossbowDataBufferReset (s->input);
		crossbowVariableReset   (s->labels);

		/* Release stream */
		crossbowThetaQueueRelease ((crossbowThetaQueueP) crossbowArrayListGet (self->streams, s->deviceId), s->id);
		dbg("Released stream\n");
	}
	self->exited = 1;
	return self;
}

crossbowCallbackHandlerP crossbowCallbackHandlerCreate (crossbowModelManagerP modelmanager, crossbowResultHandlerP resulthandler, crossbowArrayListP streams, int core) {
	crossbowCallbackHandlerP p;
	p = (crossbowCallbackHandlerP) crossbowMalloc (sizeof(crossbow_callbackhandler_t));
	p->exit = 0;
	p->exited = 0;
	p->id = autoincrement++;
	/* Create a barrier between the main thread and a callback handler */
	pthread_barrier_init(&(p->barrier), NULL, 2);
	/* Only 1 event should suffice */
    p->events = crossbowWaitFreeQueueCreate(10);
    
	/* Pointers to execution context's members */
	p->modelmanager = modelmanager;
	p->resulthandler = resulthandler;
	p->streams = streams;
    
    p->core = core;
    
#ifdef INTRA_TASK_MEASUREMENTS
	p->measurements = crossbowMeasurementListCreate (64, 0);
#endif
	
    /* Launch thread */
	pthread_create (&(p->thread), NULL, handle, (void *) p);
	/* Wait for callback handler to be initialised */
	pthread_barrier_wait (&(p->barrier));
	return p;
}

void crossbowCallbackHandlerPublish (crossbowCallbackHandlerP p, crossbowStreamP args) {
	/* Should we wait for a slot to be freed? */
    crossbowWaitFreeQueueEnqueueOrWait(p->events, args);
}

void crossbowCallbackHandlerFree (crossbowCallbackHandlerP p) {
	if (! p->exited) {
        p->exit = 1;
        /* Unblock thread */
        crossbowWaitFreeQueueEnqueue(p->events, NULL);
    }
    /* Wait until thread has exited */
    pthread_join(p->thread, NULL);
    /* Queue of events should be empty */
	crossbowWaitFreeQueueFree (p->events);
	/* Clean-up pthread_* */
	pthread_barrier_destroy (&(p->barrier));
#ifdef INTRA_TASK_MEASUREMENTS
	crossbowMeasurementListFree(p->measurements);
#endif
	crossbowFree (p, sizeof(crossbow_callbackhandler_t));
}
