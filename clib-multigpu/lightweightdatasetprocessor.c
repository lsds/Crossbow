#include "lightweightdatasetprocessor.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <pthread.h>

#define FREE_LIST_STASH 128

#define TYPES 2

static unsigned long autoincrement = 0UL;

static void *handle (void *args) {

	crossbowLightWeightDatasetProcessorP     self = NULL;
	crossbowLightWeightDatasetProcessorTaskP task = NULL;

	crossbowLightWeightDatasetTaskP p;
	int i = 0;

	self = (crossbowLightWeightDatasetProcessorP) args;

	if (self->offset > 0) {
		cpu_set_t set;
		int core = self->offset + ((int) (self->id));
		CPU_ZERO (&set);
		CPU_SET  (core, &set);
		sched_setaffinity (0, sizeof(set), &set);
		info("Light-weight dataset handler #%lu pinned on core %2d\n", self->id, core);
	}

	pthread_mutex_lock(&(self->lock));

	dbg("Light-weight data set handler #%lu starts\n", self->id);

	while (! self->exit) {
		/* Block, waiting for an event */
		while (!self->exit && crossbowListSize(self->events) == 0) {
			pthread_cond_wait(&(self->cond), &(self->lock));
		}
		if (self->exit) {
			break;
		}
		/* Get task */
		task = (crossbowLightWeightDatasetProcessorTaskP) crossbowListRemoveFirst (self->events);

		if (task->op == RESERVE) {

			/* 0 for examples; 1 for labels */
			for (i = 0; i < TYPES; ++i) {

				/* Get the current task data */
				p = &(task->slot[i]->table[task->slot[i]->ndx]);

				if (p->offset == 0) {
					/* This is the first task associated with p->file[0]. */
					crossbowDatasetFileAdviceWillNeed (p->file[0]);
				}
				if (p->file[1]) {
					/* This is the first task associated with p->file[1] */
					crossbowDatasetFileAdviceWillNeed (p->file[1]);
				}

				/* Copy data into the buffer */
				if (p->file[1]) {
					/* Copy in two parts */
					int left  = p->file[0]->length - p->offset;
					int right = p->length - left;

					/* Copy 1st part */
					crossbowLightWeightDatasetBufferCopy (
						task->slot[i]->buffer,
						task->slot[i]->offset,
						/* Source buffer is the first file */
						p->file[0]->data,
						p->offset,
						left);

					/* Copy 2nd part */
					crossbowLightWeightDatasetBufferCopy (
						task->slot[i]->buffer,
						task->slot[i]->offset + left,
						/* Source buffer is the second file */
						p->file[1]->data,
						0,
						right);
				} else {
					/* Data resides on a single file */
					crossbowLightWeightDatasetBufferCopy (
						task->slot[i]->buffer,
						task->slot[i]->offset,
						/* Source buffer is the file */
						p->file[0]->data,
						p->offset,
						p->length);
				}

				/* Set next task */
				task->slot[i]->ndx = (task->slot[i]->ndx + task->slot[i]->inc) % task->slot[i]->max;
			}

			/* Reserve slot (examples and labels share the same handler) */
			crossbowLightWeightDatasetHandlerReserve (task->handler, task->phi, task->slot[0]->id);
		}
		else {

			for (i = 0; i < TYPES; ++i) {
				/* Get the current task data */
				p = &(task->slot[i]->table[task->slot[i]->ndx]);

				if ((p->file[1]) || ((p->offset + p->length) == p->file[0]->length)) {
					/* This is the last task associated with p->file[0] */
					crossbowDatasetFileAdviceDontNeed (p->file[0]);
				}
			}
			/* Release slot (examples and labels share the same handler) */
			crossbowLightWeightDatasetHandlerRelease (task->handler, task->phi, task->slot[0]->id);
		}

		/* Return task to free list */
		crossbowLightWeightDatasetProcessorPutTaskSafely (self, task);
	}
	self->exited = 1;
	return self;
}

crossbowLightWeightDatasetProcessorP crossbowLightWeightDatasetProcessorCreate (int offset) {
	crossbowLightWeightDatasetProcessorP p;
	p = (crossbowLightWeightDatasetProcessorP) crossbowMalloc (sizeof(crossbow_lightweightdatasetprocessor_t));
	memset (p, 0, sizeof(crossbow_lightweightdatasetprocessor_t));
	p->exit = 0;
	p->exited = 0;
	p->id = autoincrement++;
	pthread_mutex_init (&(p->lock), NULL);
	pthread_cond_init  (&(p->cond), NULL);
	p->events = crossbowListCreate ();
	p->offset = offset;
	/* Manage free list */
	pthread_mutex_init (&(p->sync), NULL);
	/* Initialise free list; p->pool and p->freeList should be NULL. */
	crossbowLightWeightDatasetProcessorCreateTaskPool (p, FREE_LIST_STASH);

	/* Launch thread */
	pthread_create (&(p->thread), NULL, handle, (void *) p);
	return p;
}

crossbowLightWeightDatasetProcessorTaskP crossbowLightWeightDatasetProcessorGetTask (crossbowLightWeightDatasetProcessorP handler) {
	crossbowLightWeightDatasetProcessorTaskP p = NULL;
	pthread_mutex_lock(&(handler->sync));
	if (! (p = handler->freeList)) {
		p = crossbowLightWeightDatasetProcessorCreateTaskPool (handler, FREE_LIST_STASH);
	}
	handler->freeList = p->next;
	pthread_mutex_unlock(&(handler->sync));
	return p;
}

crossbowLightWeightDatasetProcessorTaskP crossbowLightWeightDatasetProcessorCreateTaskPool (crossbowLightWeightDatasetProcessorP handler, int size) {
	int i;
	crossbowLightWeightDatasetProcessorTaskP task;
	crossbowLightWeightDatasetProcessorTaskPoolP pool;
	/* Create a new pool only when the list of free tasks is empty */
	invalidConditionException (handler->freeList == NULL);
	/* Create a new stash of tasks */
	task = (crossbowLightWeightDatasetProcessorTaskP) crossbowMalloc (size * sizeof(crossbow_lightweightdatasetprocessortask_t));
	/* Create a new pool and initialise it */
	pool = (crossbowLightWeightDatasetProcessorTaskPoolP) crossbowMalloc (sizeof(crossbow_lightweightdatasetprocessortaskpool_t));
	pool->task = task;
	pool->size = size;
	pool->next = handler->pool;
	handler->pool = pool;
	/* Append new nodes to free list */
	for (i = 0; i < pool->size; i++, task++)
		crossbowLightWeightDatasetProcessorPutTask (handler, task);
	return handler->freeList;
}

void crossbowLightWeightDatasetProcessorPutTaskSafely (crossbowLightWeightDatasetProcessorP handler, crossbowLightWeightDatasetProcessorTaskP p) {
	pthread_mutex_lock(&(handler->sync));
	/* Reset task here */
	p->slot[0] = NULL;
	p->slot[1] = NULL;
	p->next = handler->freeList;
	handler->freeList = p;
	pthread_mutex_unlock(&(handler->sync));
	return;
}

void crossbowLightWeightDatasetProcessorPutTask (crossbowLightWeightDatasetProcessorP handler, crossbowLightWeightDatasetProcessorTaskP p) {
	/* Reset task here */
	p->slot[0] = NULL;
	p->slot[1] = NULL;
	p->next = handler->freeList;
	handler->freeList = p;
	return;
}

void crossbowLightWeightDatasetProcessorPublish (crossbowLightWeightDatasetProcessorP p, crossbowLightWeightDatasetProcessorTaskP args) {
	pthread_mutex_lock (&(p->lock));
	if (! p->exit) {
		crossbowListAppend (p->events, args);
		pthread_cond_signal (&(p->cond));
	}
	pthread_mutex_unlock(&(p->lock));
}

void crossbowLightWeightDatasetProcessorFree (crossbowLightWeightDatasetProcessorP p) {
	pthread_mutex_lock(&(p->lock));
	if (! p->exited) {
		p->exit = 1;
		pthread_cond_signal (&(p->cond));
	}
	pthread_mutex_unlock(&(p->lock));
	/* Wait until thread has exited */
	pthread_join(p->thread, NULL);
	/* Free pool of nodes */
	nullPointerException(p->freeList);
	crossbowLightWeightDatasetProcessorTaskP last = p->freeList;
	int available = 1;
	while (last->next != NULL) {
		last = last->next;
		available++;
	}
	/* There exists at least one pool of tasks */
	nullPointerException (p->pool);
	crossbowLightWeightDatasetProcessorTaskPoolP temp, pool = p->pool;
	int allocated = 0;
	while (pool != NULL) {
		temp = pool;
		pool = pool->next;
		/* Increment counter */
		allocated += temp->size;
		/* Free `temp` */
		crossbowFree (temp->task, temp->size * sizeof(crossbow_lightweightdatasetprocessortask_t));
		crossbowFree (temp, sizeof(crossbow_lightweightdatasetprocessortaskpool_t));
	}
	dbg("%2d/%2d nodes in pool\n", available, allocated);
	invalidConditionException(available == allocated);
	/* List should be empty */
	crossbowListFree (p->events);
	crossbowFree (p, sizeof(crossbow_lightweightdatasetprocessor_t));
}
