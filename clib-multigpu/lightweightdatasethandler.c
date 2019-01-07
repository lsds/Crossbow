#include "lightweightdatasethandler.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <stdio.h>

/* 2 result handlers, one for training and one for test tasks */
#define NR 2

crossbowLightWeightDatasetHandlerP crossbowLightWeightDatasetHandlerCreate (int pad) {
	int i;
	crossbowLightWeightDatasetHandlerP p;
	p = (crossbowLightWeightDatasetHandlerP) crossbowMalloc (sizeof(crossbow_lightweightdatasethandler_t));
	p->slots = (void **) crossbowMalloc (NR * sizeof(void *));
	p->count = (int *)   crossbowMalloc (NR * sizeof(int)   );
	for (i = 0; i < NR; i++) {
		p->slots[i] = NULL;
		p->count[i] = 0;
	}
	p->offset = pad;
	return p;
}

void  crossbowLightWeightDatasetHandlerSet (crossbowLightWeightDatasetHandlerP p, int id, void *buffer, int count) {
	indexOutOfBoundsException (id, NR);
	p->slots[id] = buffer;
	p->count[id] =  count;
	return;
}

void *crossbowLightWeightDatasetHandlerGet (crossbowLightWeightDatasetHandlerP p, int id) {
	indexOutOfBoundsException (id, NR);
	return p->slots[id];
}

int crossbowLightWeightDatasetHandlerNumberOfSlots (crossbowLightWeightDatasetHandlerP p, int id) {
	indexOutOfBoundsException (id, NR);
	return p->count[id];
}

void crossbowLightWeightDatasetHandlerCompareAndSwap (crossbowLightWeightDatasetHandlerP p, int phi, int ndx, int comp, int exch) {
	void *slots;
	int  offset;
	slots = crossbowLightWeightDatasetHandlerGet (p, phi);
	offset = ndx * p->offset;
	while (comp != __sync_val_compare_and_swap ((int *)(slots + offset), comp, exch))
		;
	return;
}

void crossbowLightWeightDatasetHandlerPrepareToReserve (crossbowLightWeightDatasetHandlerP p, int phi, int ndx) {

	crossbowLightWeightDatasetHandlerCompareAndSwap (p, phi, ndx, 0, 1);
}

void crossbowLightWeightDatasetHandlerReserve (crossbowLightWeightDatasetHandlerP p, int phi, int ndx) {

	crossbowLightWeightDatasetHandlerCompareAndSwap (p, phi, ndx, 1, 2);
}

void crossbowLightWeightDatasetHandlerReady (crossbowLightWeightDatasetHandlerP p, int phi, int ndx) {

	crossbowLightWeightDatasetHandlerCompareAndSwap (p, phi, ndx, 2, 3);
}

void crossbowLightWeightDatasetHandlerPrepareToRelease (crossbowLightWeightDatasetHandlerP p, int phi, int ndx) {

	crossbowLightWeightDatasetHandlerCompareAndSwap (p, phi, ndx, 3, 4);
}

void crossbowLightWeightDatasetHandlerRelease (crossbowLightWeightDatasetHandlerP p, int phi, int ndx) {

	crossbowLightWeightDatasetHandlerCompareAndSwap (p, phi, ndx, 4, 0);
}

void crossbowLightWeightDatasetHandlerFree (crossbowLightWeightDatasetHandlerP p) {
	crossbowFree (p->slots, NR * sizeof(void *));
	crossbowFree (p->count, NR * sizeof(int)   );
	crossbowFree (p, sizeof(crossbow_lightweightdatasethandler_t));
	return;
}
