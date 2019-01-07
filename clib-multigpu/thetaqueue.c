#include "thetaqueue.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#define THETA_QUEUE_SLOT_SIZE 64
#define THETA_QUEUE_SLOT_FREE  0
#define THETA_QUEUE_SLOT_BUSY  1
#define THETA_QUEUE_SLOT_SKIP  2

crossbowThetaQueueP crossbowThetaQueueCreate (int size) {
	int i;
	crossbowThetaQueueP q;
	q = (crossbowThetaQueueP) crossbowMalloc (sizeof(crossbow_theta_queue_t));
	q->size = size;
	q->iter = -1; /* Next iterator is element at position 0 */

	q->slots = (void *) crossbowMallocAligned (4096, q->size * THETA_QUEUE_SLOT_SIZE);
	memset (q->slots, 0, q->size * THETA_QUEUE_SLOT_SIZE);

	q->elements = crossbowMalloc (q->size * sizeof(void *));
	for (i = 0; i < q->size; ++i)
		q->elements[i] = NULL;

	return q;
}

/* Add new element is the last slot */
void crossbowThetaQueueExpand (crossbowThetaQueueP q, void *elem) {
	int i;
	int size_ = q->size + 1;
	void *slots_ = (void *) crossbowMallocAligned (4096, size_ * THETA_QUEUE_SLOT_SIZE);
	memset (slots_, 0, size_ * THETA_QUEUE_SLOT_SIZE);
	/* Copy existing contents */
	memcpy (slots_, q->slots, q->size * THETA_QUEUE_SLOT_SIZE);

	void **elements_ = crossbowMalloc (size_ * sizeof(void *));
	for (i = 0; i < q->size; ++i)
		elements_[i] = q->elements[i];
	elements_[size_ - 1] = elem;
    /*
	dbg("Added %p. New theta contains:\n", elem);
	for (i = 0; i < size_; i++) {
		info("%2d: %p\n", i, elements_[i]);
	}
    */
	/* Swap */
	q->size = size_;
	/* dbg("New size is %d\n", q->size); */
	q->slots = slots_;
	q->elements = elements_;
	/* crossbowThetaQueueFree(t); */
	return;
}

/* Simply disable the element */
int crossbowThetaQueueShrink (crossbowThetaQueueP q, void *elem) {
	/* Find index of element */
	int ndx = -1;
	int i;
	/* dbg("Find %p:\n", elem); */
	for (i = 0; i < q->size; ++i) {
		/* dbg("%2d: %p\n", i, q->elements[i]); */
		if (q->elements[i] == elem) {
			ndx = i;
			break;
		}
	}
	indexOutOfBoundsException (ndx, q->size);
	crossbowThetaQueueDisable (q, ndx);
	return 0;
}

int crossbowThetaQueueSize (crossbowThetaQueueP q) {
	nullPointerException (q);
	return q->size;
}

void crossbowThetaQueueSet (crossbowThetaQueueP q, int ndx, void *elem) {
	nullPointerException (q);
	indexOutOfBoundsException (ndx, q->size);
	q->elements[ndx] = elem;
	return;
}

void *crossbowThetaQueueGet (crossbowThetaQueueP q, int ndx) {
	nullPointerException(q);
	/* dbg("ThetaQueueGet(%d)\n", ndx); */
	indexOutOfBoundsException (ndx, q->size);
	crossbowThetaQueueReserve (q, ndx);
	return q->elements[ndx];
}

void *crossbowThetaQueueGetNext (crossbowThetaQueueP q) {
	int next;
	while (1) {
		next = (++q->iter) % q->size;
		if (crossbowThetaQueueIsEnabled(q, next)) {
			break;
		}
	}
	return crossbowThetaQueueGet (q, next);
}

void *crossbowThetaQueueGetNextSafely (crossbowThetaQueueP q) {
	int next;
	/* Atomically increment counter */
	while (1) {
		next = __sync_add_and_fetch (&(q->iter), 1);
		next %= q->size;
		if (crossbowThetaQueueIsEnabled(q, next))
			break;
	}
	return crossbowThetaQueueGet (q, next);
}

void crossbowThetaQueueReserve (crossbowThetaQueueP q, int ndx) {

	unsigned long long spins = 0ULL;

	int offset = THETA_QUEUE_SLOT_SIZE * ndx;

	int comp   = THETA_QUEUE_SLOT_FREE;
	int exch   = THETA_QUEUE_SLOT_BUSY;

	while (comp != __sync_val_compare_and_swap ((int *)(q->slots + offset), comp, exch)) {
		/* info("Spin at %d: current value is %d\n", ndx, *(int *)(q->slots + offset)); */
		spins ++;
	}

	return;
}

void crossbowThetaQueueRelease (crossbowThetaQueueP q, int ndx) {

	unsigned long long spins = 0ULL;

	int offset = THETA_QUEUE_SLOT_SIZE * ndx;

	int comp   = THETA_QUEUE_SLOT_BUSY;
	int exch   = THETA_QUEUE_SLOT_FREE;

	while (comp != __sync_val_compare_and_swap ((int *)(q->slots + offset), comp, exch)) {
		spins ++;
	}

	return;
}

int crossbowThetaQueueIsEnabled (crossbowThetaQueueP q, int ndx) {

	int offset = THETA_QUEUE_SLOT_SIZE * ndx;
	int comp   = THETA_QUEUE_SLOT_SKIP;

	return (comp != *(int *)(q->slots + offset));
}

void crossbowThetaQueueEnable (crossbowThetaQueueP q, int ndx) {

	int offset = THETA_QUEUE_SLOT_SIZE * ndx;

	int comp   = THETA_QUEUE_SLOT_SKIP;
	int exch   = THETA_QUEUE_SLOT_FREE;

	if (comp != __sync_val_compare_and_swap ((int *)(q->slots + offset), comp, exch)) {
		err("Invalid state\n");
	}

	return;
}

int crossbowThetaQueueIsDisabled (crossbowThetaQueueP q, int ndx) {

	int offset = THETA_QUEUE_SLOT_SIZE * ndx;
	int comp   = THETA_QUEUE_SLOT_SKIP;

	return (comp == *(int *)(q->slots + offset));
}

int crossbowThetaQueueDisable (crossbowThetaQueueP q, int ndx) {

	int offset = THETA_QUEUE_SLOT_SIZE * ndx;

	int comp   = THETA_QUEUE_SLOT_FREE;
	int exch   = THETA_QUEUE_SLOT_SKIP;

	if (comp != __sync_val_compare_and_swap ((int *)(q->slots + offset), comp, exch)) {
		return 1;
	}
	info("Disabled object #%d\n", ndx);
	return 0;
}

int crossbowThetaQueueDisableAny (crossbowThetaQueueP q) {
	int ndx;
	int enabled = 0;
	for (ndx = q->size / 2; ndx < q->size; ++ndx) {
		enabled += crossbowThetaQueueDisable (q, ndx);
	}
	// info("%d/%d objects still enabled\n", enabled, q->size);
	// if (enabled == 0) {
	//	/* Enable at least one object */
	//	info("Re-enabling object #0\n");
	//	crossbowThetaQueueEnable (q, 0);
	// }
	return 1; // enabled;
}

/*
 * This call is unsafe. Assumes that all items in the
 * queue have been dequeued.
 */
void crossbowThetaQueueFree (crossbowThetaQueueP q) {
	if (! q)
		return;
	crossbowFree (q->slots, q->size * THETA_QUEUE_SLOT_SIZE);
	crossbowFree (q->elements, q->size * sizeof(void *));
	crossbowFree (q, sizeof(crossbow_theta_queue_t));
	return;
}
