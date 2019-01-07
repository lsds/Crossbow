#include "bytebuffer.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowByteBufferP crossbowByteBufferCreate (int size) {
	crossbowByteBufferP p;
	p = (crossbowByteBufferP) crossbowMalloc (sizeof(crossbow_bytebuffer_t));
	p->size = size;
	p->data = (void *) crossbowMalloc (p->size);
	memset (p->data, 0, p->size);
	return p;
}

void crossbowByteBufferClear (crossbowByteBufferP p) {
	if (! p)
		return;
	memset (p->data, 0, p->size);
	return;
}

int crossbowByteBufferSize (crossbowByteBufferP p) {
	if (! p)
		return 0;
	return p->size;
}

void *crossbowByteBufferData (crossbowByteBufferP p) {
	if (! p)
		return NULL;
	return p->data;
}

static inline float __bswapfp (float _x) {
	float _y;
	char *x = (char *) &_x;
	char *y = (char *) &_y;
	y[0] = x[3];
	y[1] = x[2];
	y[2] = x[1];
	y[3] = x[0];
	return _y;
}

void crossbowByteBufferSwap (crossbowByteBufferP p, int count) {
	int i;
	float *f = (float *) p->data;
	for (i = 0; i < count; ++i)
		f[i] = __bswapfp (f[i]);
	return;
}

void crossbowByteBufferFree (crossbowByteBufferP p) {
	crossbowFree (p->data, p->size);
	crossbowFree (p, sizeof(crossbow_bytebuffer_t));
	return;
}
