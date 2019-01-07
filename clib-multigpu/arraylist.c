#include "arraylist.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowArrayListP crossbowArrayListCreate (int size) {
	int i;
	crossbowArrayListP p;
	p = (crossbowArrayListP) crossbowMalloc (sizeof(crossbow_arraylist_t));
	p->size = (size > 0) ? size : 1;
	p->iter = -1; /* Next points to element at position 0 */
	p->elements = crossbowMalloc (p->size * sizeof(void *));
	for (i = 0; i < p->size; ++i)
		p->elements[i] = NULL;
	return p;
}

int crossbowArrayListSize (crossbowArrayListP p) {
	return p->size;
}

void crossbowArrayListResize (crossbowArrayListP p, int newsize) {
	int i;
	void **other = crossbowMalloc (newsize * sizeof(void *));
	for (i = 0; i < newsize; ++i)
		if (i < p->size)
			other[i] = p->elements[i];
		else
			other[i] = NULL;
	void **t = p->elements;
	p->elements = other;
	crossbowFree (t, p->size * sizeof(void *));
	p->size = newsize;
	return;
}

void *crossbowArrayListGet (crossbowArrayListP p, int ndx) {
	nullPointerException(p);
	indexOutOfBoundsException (ndx, p->size);
	return p->elements[ndx];
}

void crossbowArrayListSet (crossbowArrayListP p, int ndx, void *elem) {
	int limit = p->size - 1;
	invalidArgumentException (ndx >= 0);
	if (ndx > limit)
		crossbowArrayListResize (p, (ndx + 1));
	p->elements[ndx] = elem;
	return;
}

void crossbowArrayListResetIterator (crossbowArrayListP p) {
	p->iter = -1;
}

void *crossbowArrayListGetNext (crossbowArrayListP p) {
	int next = (++p->iter) % p->size;
	return crossbowArrayListGet (p, next);
}

void *crossbowArrayListGetNextSafely (crossbowArrayListP p) {
	int next;
	/* Atomically increment counter */
	next = __sync_add_and_fetch (&(p->iter), 1);
	next %= p->size;
	return crossbowArrayListGet (p, next);
}

void crossbowArrayListFree (crossbowArrayListP p) {
	crossbowFree (p->elements, p->size * sizeof(void *));
	crossbowFree (p, sizeof(crossbow_arraylist_t));
	return;
}
