#include "resulthandler.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <stdio.h>

/* 2 result handlers, one for training and one for test tasks */
#define NR 2

crossbowResultHandlerP crossbowResultHandlerCreate (int pad) {
	int i;
	crossbowResultHandlerP p;
	p = (crossbowResultHandlerP) crossbowMalloc (sizeof(crossbow_resulthandler_t));
	p->slots = (void **) crossbowMalloc (NR * sizeof(void *));
	p->count = (int *)   crossbowMalloc (NR * sizeof(int)   );
	for (i = 0; i < NR; i++) {
		p->slots[i] = NULL;
		p->count[i] = 0;
	}
	p->offset = pad;
	return p;
}

void crossbowResultHandlerSet (crossbowResultHandlerP p, int id, void *buffer, int count) {
	indexOutOfBoundsException (id, NR);
	p->slots[id] = buffer;
	p->count[id] =  count;
	return;
}

int crossbowResultHandlerNumberOfSlots (crossbowResultHandlerP p, int id) {
	indexOutOfBoundsException (id, NR);
	return p->count[id];
}

void *crossbowResultHandlerGet (crossbowResultHandlerP p, int id) {
	indexOutOfBoundsException (id, NR);
	return p->slots[id];
}

void crossbowResultHandlerReserveSlot (crossbowResultHandlerP p, int phi, int taskid, long *freeP, float loss, float accuracy) {
	void *slots;
	int ndx, offset;
	int comp, exch;

	slots = crossbowResultHandlerGet (p, phi);
	ndx = ((taskid - 1) % crossbowResultHandlerNumberOfSlots(p, phi));
	offset = ndx * p->offset;

	comp = 0;
	exch = 1;
	while (comp != __sync_val_compare_and_swap ((int *)(slots + offset), comp, exch)) {
		/*
		fprintf(stderr, "warning: GPU result handler blocked at task %d (index %d)\n", taskid, ndx);
		fflush (stderr);
		*/
	}

	*((long *) (slots + offset +  4)) = freeP[0];
	*((long *) (slots + offset + 12)) = freeP[1];

	/* Set loss at offset +12 (after lock and the two free pointers) */
	*((float *) (slots + offset + 20)) = loss;
	*((float *) (slots + offset + 24)) = accuracy;

	comp = 1;
	exch = 2;
	if (comp != __sync_val_compare_and_swap ((int *)(slots + offset), comp, exch)) {
		fprintf(stderr, "error: failed to set slots %d (@%d) to 1\n", ndx, offset);
		exit (1);
	}
	return;
}

void crossbowResultHandlerFree (crossbowResultHandlerP p) {
	crossbowFree (p->slots, NR * sizeof(void *));
	crossbowFree (p->count, NR * sizeof(int)   );
	crossbowFree (p, sizeof(crossbow_resulthandler_t));
	return;
}
