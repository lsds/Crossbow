#ifndef __CROSSBOW_WAIT_FREE_QUEUE_H_
#define __CROSSBOW_WAIT_FREE_QUEUE_H_

typedef struct crossbow_waitfree_queue *crossbowWaitFreeQueueP;
typedef struct crossbow_waitfree_queue {
	int capacity;
	void **items;
	int head, tail;
} crossbow_waitfree_queue_t;

crossbowWaitFreeQueueP crossbowWaitFreeQueueCreate (int);

int crossbowWaitFreeQueueCapacity (crossbowWaitFreeQueueP);

void *crossbowWaitFreeQueueDequeue (crossbowWaitFreeQueueP);

void *crossbowWaitFreeQueueDequeueOrWait (crossbowWaitFreeQueueP);

void crossbowWaitFreeQueueEnqueue (crossbowWaitFreeQueueP, void *);

void crossbowWaitFreeQueueEnqueueOrWait (crossbowWaitFreeQueueP, void *);

int crossbowWaitFreeQueueEmpty (crossbowWaitFreeQueueP);

int crossbowWaitFreeQueueFull (crossbowWaitFreeQueueP);

void crossbowWaitFreeQueueFree (crossbowWaitFreeQueueP);

#endif /* __CROSSBOW_WAIT_FREE_QUEUE_H_ */
