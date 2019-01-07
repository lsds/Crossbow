#ifndef __CROSSBOW_THREAD_SAFE_QUEUE_H_
#define __CROSSBOW_THREAD_SAFE_QUEUE_H_

#include <pthread.h>

#include "listnode.h"

#include "listnodepool.h"

#include "utils.h"

typedef struct crossbow_threadsafe_queue *crossbowThreadSafeQueueP;
typedef struct crossbow_threadsafe_queue {
	crossbowListNodeP head;
	crossbowListNodeP tail;
	int size;
	int capacity;
	crossbowListNodeP freeList;
	crossbowListNodePoolP pool;

	pthread_mutex_t lock;

	pthread_cond_t hassome; /* At least one element exists in the queue */
	pthread_cond_t notfull; /* At least one free slot exists */

} crossbow_threadsafe_queue_t;

crossbowThreadSafeQueueP crossbowThreadSafeQueueCreate (int);

int crossbowThreadSafeQueueCapacity (crossbowThreadSafeQueueP);

int crossbowThreadSafeQueueSize (crossbowThreadSafeQueueP);

void *crossbowThreadSafeQueueDequeue (crossbowThreadSafeQueueP);

void *crossbowThreadSafeQueueDequeueOrWait (crossbowThreadSafeQueueP);

void crossbowThreadSafeQueueEnqueue (crossbowThreadSafeQueueP, void *);

void crossbowThreadSafeQueueExpand (crossbowThreadSafeQueueP, void *);

void crossbowThreadSafeQueueExpandVirtually (crossbowThreadSafeQueueP);

int crossbowThreadSafeQueueShrink (crossbowThreadSafeQueueP, void *);

int crossbowThreadSafeQueueEmpty (crossbowThreadSafeQueueP);

int crossbowThreadSafeQueueFull (crossbowThreadSafeQueueP);

char *crossbowThreadSafeQueueString (crossbowThreadSafeQueueP, crossbowListNodeItem_t);

void crossbowThreadSafeQueueFree (crossbowThreadSafeQueueP);

/* Thread-unsafe queue iterator */

crossbowListNodeP crossbowThreadSafeQueueIteratorCreate (crossbowThreadSafeQueueP);

void *crossbowThreadSafeQueueIteratorNext (crossbowListNodeP);

void crossbowThreadSafeQueueIteratorFree (crossbowListNodeP);

#endif /* __CROSSBOW_THREAD_SAFE_QUEUE_H_ */
