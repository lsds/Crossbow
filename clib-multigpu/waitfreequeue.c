#include "waitfreequeue.h"

/*
 * Based on http://www.read.seas.harvard.edu/~kohler/class/05s-osp/notes/lec7.c
 */

#include "debug.h"
#include "utils.h"

#include  <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <sched.h>

crossbowWaitFreeQueueP crossbowWaitFreeQueueCreate (int capacity) {
	int i;
	crossbowWaitFreeQueueP q = (crossbowWaitFreeQueueP) malloc (sizeof(crossbow_waitfree_queue_t));
	if (! q) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
	q->capacity = capacity;
	q->items = (void **) malloc ((q->capacity + 1) * sizeof (void *));
	if (! q->items) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
	for (i = 0; i < q->capacity; i++)
		q->items[i] = NULL;
	q->head = 0;
	q->tail = 0;
	return q;
}

int crossbowWaitFreeQueueCapacity (crossbowWaitFreeQueueP q) {
	return q->capacity;
}

void *crossbowWaitFreeQueueDequeue (crossbowWaitFreeQueueP q) {
	void *item;
	if (crossbowWaitFreeQueueEmpty(q))
		return NULL;
	item = q->items[q->head];
	q->head = (q->head + 1) % (q->capacity + 1);
	return item;
}

void *crossbowWaitFreeQueueDequeueOrWait (crossbowWaitFreeQueueP q) {
	void *item;
	while (crossbowWaitFreeQueueEmpty(q))
		; /* Alternatives are sched_yield() or pthread_yield(); */
	item = q->items[q->head];
	q->head = (q->head + 1) % (q->capacity + 1);
	return item;
}

void crossbowWaitFreeQueueEnqueue (crossbowWaitFreeQueueP q, void *item) {
	if (crossbowWaitFreeQueueFull(q)) {
		fprintf(stderr, "error: wait-free queue is full\n");
		exit (1);
	}
	q->items[q->tail] = item;
	q->tail = (q->tail + 1) % (q->capacity + 1);
	return;
}

void crossbowWaitFreeQueueEnqueueOrWait (crossbowWaitFreeQueueP q, void *item) {
	while (crossbowWaitFreeQueueFull(q)) {
		; 
	}
	q->items[q->tail] = item;
	q->tail = (q->tail + 1) % (q->capacity + 1);
	return;
}

int crossbowWaitFreeQueueEmpty (crossbowWaitFreeQueueP q) {
	return (q->head == q->tail);
}

int crossbowWaitFreeQueueFull (crossbowWaitFreeQueueP q) {
	return ((q->tail + 1) % (q->capacity + 1) == q->head);
}

void crossbowWaitFreeQueueFree (crossbowWaitFreeQueueP q) {
	free (q->items);
	free (q);
	return;
}
