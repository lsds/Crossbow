#include "bufferpool.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowBufferPoolP crossbowBufferPoolCreate (int size, int capacity) {
	int i;
	crossbowBufferPoolP p;
	p = (crossbowBufferPoolP) crossbowMalloc (sizeof(crossbow_bufferpool_t));
	p->size = size;
	p->buffers = (crossbowByteBufferP *) crossbowMalloc (p->size * sizeof(crossbowByteBufferP));
	p->capacity = capacity;
	for (i = 0; i < p->size; i++)
		p->buffers[i] = NULL;
	return p;
}

crossbowByteBufferP crossbowBufferPoolGet (crossbowBufferPoolP pool, int ndx) {
	crossbowByteBufferP p;
	indexOutOfBoundsException (ndx, pool->size);
	p = pool->buffers[ndx];
	/* Lazy materialisation */
	if (! p)
		return crossbowByteBufferCreate (pool->capacity);
	return p;
}

void crossbowBufferPoolRelease (crossbowBufferPoolP pool, int ndx, crossbowByteBufferP p) {
	indexOutOfBoundsException (ndx, pool->size);
	pool->buffers[ndx] = p;
	return;
}

int crossbowBufferPoolSize (crossbowBufferPoolP pool) {
	return pool->size;
}

void crossbowBufferPoolFree (crossbowBufferPoolP pool) {
	int i;
	for (i = 0; i < pool->size; i++)
		if (pool->buffers[i])
			crossbowByteBufferFree (pool->buffers[i]);
	crossbowFree (pool->buffers, pool->size * sizeof(crossbowByteBufferP));
	crossbowFree (pool, sizeof(crossbow_bufferpool_t));
	return;
}


