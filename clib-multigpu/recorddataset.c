#include "recorddataset.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

static void *handle (void *args) {

	crossbowRecordDatasetP      self  = NULL;
	crossbowRecordDatasetEventP event = NULL;

	self = (crossbowRecordDatasetP) args;

	cpu_set_t set;
	int core = 18;
	CPU_ZERO (&set);
	CPU_SET  (core, &set);
	sched_setaffinity (0, sizeof(set), &set);
	info("Record dataset handler pinned on core %d\n", core);

	pthread_mutex_lock(&(self->lock));

	dbg("Record data set worker starts\n");

	while (! self->exit) {
		/* Block, waiting for an event */
		while (!self->exit && crossbowListSize(self->events) == 0) {
			pthread_cond_wait(&(self->cond), &(self->lock));
		}
		if (self->exit) {
			break;
		}
		
		/* Get task */
		event = (crossbowRecordDatasetEventP) crossbowListRemoveFirst (self->events);

		dbg("New task: fill %d\n", event->idx);

		/* Process event */

		crossbowRecordReaderReadProperly (self->reader,
				self->phi == TRAIN ? 1 : 0,
				self->count,
				self->buffer->size,
				self->buffer->b,
				self->buffer->padding,
				self->buffer->theImages[event->idx],
				self->buffer->theLabels[event->idx],
				self->buffer->capacity
		);

		crossbowDoubleBufferUnlock (self->buffer, event->idx);

		/* Return task to free list */
		crossbowFree (event, sizeof(crossbow_record_dataset_event_t));
	}
	info("Record data set worker exits\n");
	
	self->exited = 1;
	return self;
}

crossbowRecordDatasetP crossbowRecordDatasetCreate (int workers, int *capacity, int NB, int b, int *padding, crossbowPhase_t phi) {

	crossbowRecordDatasetP p = (crossbowRecordDatasetP) crossbowMalloc (sizeof(crossbow_record_dataset_t));
	
	/* Distinguish between training and validation datasets */
	p->phi = phi;

	p->reader = crossbowRecordReaderCreate (workers, (p->phi == CHECK) ? 0 : 1);

	p->buffer = crossbowDoubleBufferCreate (capacity, NB, b, padding);
	crossbowDoubleBufferRegister       (p->buffer);
	crossbowDoubleBufferAdviceWillNeed (p->buffer);
	
	/* Current data pointers */
	p->images = NULL;
	p->labels = NULL;
	
	/* Number of images and labels to decode/read per read call */
	p->count = (NB * b);
	
	p->latch = 0;
	
	/* Set-up worker thread */
	p->exit = 0;
	p->exited = 0;

	pthread_mutex_init (&(p->lock), NULL);
	pthread_cond_init  (&(p->cond), NULL);

	p->events = crossbowListCreate ();
	

	/* Launch thread */
	pthread_create (&(p->thread), NULL, handle, (void *) p);

	return p;
}

void crossbowRecordDatasetInitSafely (crossbowRecordDatasetP p) {

	nullPointerException (p);

	invalidConditionException (p->images == NULL);
	invalidConditionException (p->labels == NULL);

	info("%d items (%d/%d bytes), %d items per batch (%d/%d bytes padding), write to %p/%p, limited to %d/%d bytes\n",
			p->count,
			p->buffer->size[0],
			p->buffer->size[1],
			p->buffer->b,
			p->buffer->padding[0],
			p->buffer->padding[1],
			p->buffer->theImages[0],
			p->buffer->theLabels[0],
			p->buffer->capacity[0],
			p->buffer->capacity[1]);

	crossbowRecordReaderReadProperly (p->reader,
					p->phi == TRAIN ? 1 : 0,
					p->count,
					p->buffer->size,
					p->buffer->b,
					p->buffer->padding,
					p->buffer->theImages[0],
					p->buffer->theLabels[0],
					p->buffer->capacity
			);

	/* Ensure that first call to `swap` will use the above buffer */
	p->buffer->idx = 1;
	crossbowDoubleBufferLock (p->buffer, 1);
	
	/* Set count-down latch to 0 (don't wait for any tasks) */
	p->latch = 0;

	return;
}

void crossbowRecordDatasetSwap (crossbowRecordDatasetP p) {

	nullPointerException (p);

	/* Increment task index */
	int prev = p->buffer->idx;
	int next = (++p->buffer->idx) % 2;
	p->buffer->idx = next;

	dbg("Swap from %d to %d\n", prev, next);

	/* Unlock previous buffer */
	crossbowDoubleBufferUnlock (p->buffer, prev);

	/* Wait until at least one buffer is filled by reader */
	crossbowDoubleBufferLock (p->buffer, next);

	/* Change buffer pointers */
	p->images = p->buffer->theImages [next];
	p->labels = p->buffer->theLabels [next];

	/* Create a new task to fill previous buffer (but lock it first) */
	crossbowDoubleBufferLock (p->buffer, prev);
	
	/* Make sure that all tasks using previous buffer have completed */
	while (p->latch != 0);
	/* Reset count-down latch for the current buffer.
	 * This should be a thread-safe operation. */
	p->latch = p->buffer->NB;

	crossbowRecordDatasetEventP event = crossbowMalloc (sizeof(crossbow_record_dataset_event_t));
	event->idx = prev;
	/* Schedule task */
	pthread_mutex_lock (&(p->lock));
	if (! p->exit) {
		crossbowListAppend (p->events, event);
		pthread_cond_signal (&(p->cond));
	}
	pthread_mutex_unlock(&(p->lock));

	return;
}

void crossbowRecordDatasetNotify (crossbowRecordDatasetP p) {
	nullPointerException(p);
	unsigned int value = __sync_sub_and_fetch (&(p->latch), 1);
#ifdef GPU_VERBOSE	
	info("Current latch value is %d\n", value);
#else
	(void) value;
#endif
	return;
}

void crossbowRecordDatasetFree (crossbowRecordDatasetP p) {
	if (! p)
		return;
	
	/* Wait for thread to exit (it may still swapping) */
	p->exit = 1;
	pthread_join(p->thread, NULL);
	
	/* Free buffer */
	info("Free double buffer\n");
	if (p->buffer)
		crossbowDoubleBufferFree (p->buffer);
	
	/* Free record dataset */
	info("Free record reader\n");
	if (p->reader)
		crossbowRecordReaderFree (p->reader);

	crossbowListFree (p->events);

	crossbowFree (p, sizeof(crossbow_record_dataset_t));
	return;
}
