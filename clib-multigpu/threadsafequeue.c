#include "threadsafequeue.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

/* Return node to free list */
static void putNode (crossbowThreadSafeQueueP queue, crossbowListNodeP p) {
   p->item = NULL;
   p->next = queue->freeList;
   queue->freeList = p;
}

static crossbowListNodeP crossbowThreadSafeQueueCreatePool (crossbowThreadSafeQueueP queue, int size) {
	int i;
	crossbowListNodeP node;
	crossbowListNodePoolP pool;
	/* Create a new pool only when the list of free nodes is empty */
	invalidConditionException (queue->freeList == NULL);
	/* Create a new stash of nodes */
	node = (crossbowListNodeP) crossbowMalloc (size * sizeof(crossbow_list_node_t));
	/* Create a new pool and initialise it */
	pool = (crossbowListNodePoolP) crossbowMalloc (sizeof(crossbow_list_node_pool_t));
	pool->node = node;
	pool->size = size;
	pool->next = queue->pool;
	queue->pool = pool;
	/* Append new nodes to free list */
	for (i = 0; i < pool->size; i++, node++)
		putNode (queue, node);
	return queue->freeList;
}

static crossbowListNodeP getNode (crossbowThreadSafeQueueP queue) {
	crossbowListNodeP p = NULL;
	/* If `freeList` is NULL, all nodes are in use. */
	if (! (p = queue->freeList)) {
		/* Create a new pool of nodes and append them to the `freeList`.
		 * By default the new pool size is 1.
		 */
		p = crossbowThreadSafeQueueCreatePool (queue, 1);
	}
	queue->freeList = p->next;
	return p;
}

crossbowThreadSafeQueueP crossbowThreadSafeQueueCreate (int capacity) {
	crossbowThreadSafeQueueP q;
	q = (crossbowThreadSafeQueueP) crossbowMalloc (sizeof(crossbow_threadsafe_queue_t));
	memset (q, 0, sizeof(crossbow_threadsafe_queue_t));

	q->capacity = capacity;

	crossbowThreadSafeQueueCreatePool (q, q->capacity);

	pthread_mutex_init(&(q->lock),    NULL);

	pthread_cond_init (&(q->hassome), NULL);
	pthread_cond_init (&(q->notfull), NULL);

	return q;
}

int crossbowThreadSafeQueueCapacity (crossbowThreadSafeQueueP q) {
	int result;
	pthread_mutex_lock(&(q->lock));
	result = q->capacity;
	pthread_mutex_unlock(&(q->lock));
	return result;
}

int crossbowThreadSafeQueueSize (crossbowThreadSafeQueueP q) {
	int result;
	pthread_mutex_lock(&(q->lock));
	result = q->size;
	pthread_mutex_unlock(&(q->lock));
	return result;
}

void *crossbowThreadSafeQueueDequeueOrWait (crossbowThreadSafeQueueP q) {
	crossbowListNodeP node;
	void *item;

	pthread_mutex_lock(&(q->lock));
	/* Wait until queue is not empty */
	while (crossbowThreadSafeQueueEmpty(q)) {
		pthread_cond_wait(&(q->hassome), &(q->lock));
	}

	/* Thread-safe code */
	node = q->head;
	item = q->head->item;
	q->head = q->head->next;
	if (! q->head)
		q->tail = NULL;
	q->size--;
	putNode(q, node);

	/* Signal thread(s) waiting for a free slot (to enqueue) */
	// pthread_cond_signal (&(q->notfull));
	/* Releasing the lock */
	pthread_mutex_unlock(&(q->lock));
	/* Signal thread(s) waiting for a free slot (to enqueue) */
	pthread_cond_signal (&(q->notfull));

	return item;
}

void *crossbowThreadSafeQueueDequeue (crossbowThreadSafeQueueP q) {
	crossbowListNodeP node;
	void *item;
	pthread_mutex_lock(&(q->lock));
	if (crossbowThreadSafeQueueEmpty(q)) {
		pthread_mutex_unlock(&(q->lock));
		return NULL;
	}
	/* Thread-safe code */
	node = q->head;
	item = q->head->item;
	q->head = q->head->next;
	if (! q->head)
		q->tail = NULL;
	q->size--;
	putNode(q, node);
	/* Signal thread(s) waiting for a free slot (to enqueue) */
	// pthread_cond_signal (&(q->notfull));
	/* Releasing the lock */
	pthread_mutex_unlock(&(q->lock));

	pthread_cond_signal (&(q->notfull));

	return item;
}

void crossbowThreadSafeQueueEnqueue (crossbowThreadSafeQueueP q, void *item) {

	pthread_mutex_lock(&(q->lock));
	/* Wait until queue is not full */
	while (crossbowThreadSafeQueueFull(q)) {
		pthread_cond_wait(&(q->notfull), &(q->lock));
	}
	/* Thread-safe code */
	crossbowListNodeP p = getNode (q);
	p->item = item;
	p->next = NULL;
	if (q->tail)
		q->tail->next = p;
	else
		q->head = p;
	q->tail = p;
	q->size++;

	/* Signal thread(s) waiting for at least one element to appear in the list */
	// pthread_cond_signal (&(q->hassome));
	/* Releasing the lock */
	pthread_mutex_unlock(&(q->lock));

	pthread_cond_signal (&(q->hassome));
	return;
}

void crossbowThreadSafeQueueExpand (crossbowThreadSafeQueueP q, void *item) {
	/* Safely increment queue's capacity */
	pthread_mutex_lock(&(q->lock));
	q->capacity ++;
	/* Thread-safe code */
	crossbowListNodeP p = getNode (q);
	p->item = item;
	p->next = NULL;
	if (q->tail)
		q->tail->next = p;
	else
		q->head = p;
	q->tail = p;
	q->size++;
	/* Signal thread(s) waiting for at least one element to appear in the list */
	pthread_cond_signal (&(q->hassome));
	/* Releasing the lock */
	pthread_mutex_unlock(&(q->lock));
	return;
}

void crossbowThreadSafeQueueExpandVirtually (crossbowThreadSafeQueueP q) {
	/* Safely increment queue's capacity */
	pthread_mutex_lock(&(q->lock));
	q->capacity ++;
	pthread_mutex_unlock(&(q->lock));
	return;
}

int crossbowThreadSafeQueueShrink (crossbowThreadSafeQueueP q, void *item) {
    crossbowListNodeP prev = NULL, node;
    int found = 0;
    pthread_mutex_lock(&(q->lock));
    if (! crossbowThreadSafeQueueEmpty(q)) {
        node = q->head;
        while (node) {
            if (node->item == item) {
                found ++;
                break;
            }
            prev = node;
            node = node->next;
        }
        if (found > 0) {
            if (! prev) {
                invalidConditionException (node == q->head);
                q->head = q->head->next;
                if (! q->head)
                    q->tail = NULL;
            } else {
                prev->next = node->next;
                /* Is the tail removed? */
                if (q->tail == node)
                    q->tail = prev;
            }
            q->size--;
            q->capacity--;
            putNode(q, node);
        }
    }
    pthread_mutex_unlock(&(q->lock));
    return found;
}

int crossbowThreadSafeQueueEmpty (crossbowThreadSafeQueueP q) {
	return (q->head == NULL);
}

int crossbowThreadSafeQueueFull (crossbowThreadSafeQueueP q) {
	return (q->size == q->capacity);
}

char *crossbowThreadSafeQueueString (crossbowThreadSafeQueueP q, crossbowListNodeItem_t type) {
	crossbowListNodeP p = NULL;
	char s [1024];
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	pthread_mutex_lock(&(q->lock));
	crossbowStringAppend (s, &offset, &remaining, "Thread-safe queue %p [", q);
	p = q->head;
	while (p != NULL) {
		switch (type) {
		case INTPTR:
			if (p->item) {
				/* Dereference pointer to an integer */
				crossbowStringAppend (s, &offset, &remaining, "%d -> ", *((int *) (p->item)));
			} else {
				crossbowStringAppend (s, &offset, &remaining, "null -> ");
			}
			break;
		default:
			/* Print pointer */
			crossbowStringAppend (s, &offset, &remaining, "%p -> ", p->item);
		}
		p = p->next;
	}
	crossbowStringAppend (s, &offset, &remaining, "null] (%d/%d element%s)", q->size, q->capacity, (q->size == 1) ? "" : "s");
	pthread_mutex_unlock(&(q->lock));
	return crossbowStringCopy (s);
}

/*
 * This call is unsafe. Assumes that all items in the
 * queue have been dequeued.
 */
void crossbowThreadSafeQueueFree (crossbowThreadSafeQueueP q) {
	nullPointerException(q);
	crossbowListNodeP last = q->freeList;
	int available = 0;
	while (last) {
		available ++;
		last = last->next;
	}
	/* There exists at least one pool of nodes */
	nullPointerException (q->pool);
	crossbowListNodePoolP temp, pool = q->pool;
	int allocated = 0;
	while (pool != NULL) {
		temp = pool;
		pool = pool->next;
		/* Increment counter */
		allocated += temp->size;
		/* Free `temp` */
		crossbowFree (temp->node, temp->size * sizeof(crossbow_list_node_t));
		crossbowFree (temp, sizeof(crossbow_list_node_pool_t));
	}
	dbg("%2d/%2d nodes in pool\n", available, allocated);
	/* invalidConditionException(available == allocated); */
	crossbowFree (q, sizeof(crossbow_threadsafe_queue_t));
	return;
}

/*
 * The remaining functions are related to a thread-unsafe iterator over the list nodes.
 */
crossbowListNodeP crossbowThreadSafeQueueIteratorCreate (crossbowThreadSafeQueueP q) {
	crossbowListNodeP p = NULL;
	p = (crossbowListNodeP) crossbowMalloc (sizeof(crossbow_list_node_t));
	p->next = q->head;
	return p;
}

void *crossbowThreadSafeQueueIteratorNext (crossbowListNodeP p) {
	void *item = NULL;
	if (p->next) {
		item = p->next->item;
		p->next = p->next->next;
	}
	return item;
}

void crossbowThreadSafeQueueIteratorFree (crossbowListNodeP p) {
	crossbowFree (p, sizeof(crossbow_list_node_t));
}
