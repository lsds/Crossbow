#include "doublebuffer.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <errno.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/mman.h> /* mlock, etc. */
#include <unistd.h>

#define DOUBLE_BUFFER_SLOT_SIZE 64
#define DOUBLE_BUFFER_SLOT_FREE  0
#define DOUBLE_BUFFER_SLOT_BUSY  1

crossbowDoubleBufferP crossbowDoubleBufferCreate (int *capacity, int NB, int b, int *padding) {

	crossbowDoubleBufferP p = NULL;
	int error;

	p = (crossbowDoubleBufferP) crossbowMalloc (sizeof(crossbow_doublebuffer_t));

    p->capacity [0] = capacity [0];
    p->capacity [1] = capacity [1];

    /* Allocate buffers for images */

    info ("Allocate buffers for images\n");

    error = posix_memalign(&(p->theImages[0]), getpagesize(), p->capacity[0]);
    if (error) {
    	fprintf(stderr, "fatal error: out of memory\n");
    	exit(1);
    }

    error = posix_memalign(&(p->theImages[1]), getpagesize(), p->capacity[0]);
    if (error) {
    	fprintf(stderr, "fatal error: out of memory\n");
        exit(1);
    }

    /* Allocate buffers for labels */

    info ("Allocate buffers for labels\n");

    error = posix_memalign(&(p->theLabels[0]), getpagesize(), p->capacity[1]);
    if (error) {
    	fprintf(stderr, "fatal error: out of memory\n");
    	exit(1);
    }

    error = posix_memalign(&(p->theLabels[1]), getpagesize(), p->capacity[1]);
    if (error) {
    	fprintf(stderr, "fatal error: out of memory\n");
        exit(1);
    }

    p->NB = NB;

    /* Batch size and padding */

    p->b = b;

    p->padding [0] = padding [0];
    p->padding [1] = padding [1];

    /* Item size */

    p->size [0] = 602112; /* i.e. `sizeof(float)` x 3 x 224 x 224 */
    p->size [1] =      4; /* i.e. `sizeof( int )` x 1 */

    p->idx = 0;

    p->needed = 0;
    p->locked = 0;

    p->slots = (void *) crossbowMallocAligned (4096, (DOUBLE_BUFFER_SLOT_SIZE + DOUBLE_BUFFER_SLOT_SIZE));
    memset (p->slots, 0, (DOUBLE_BUFFER_SLOT_SIZE + DOUBLE_BUFFER_SLOT_SIZE));

    return p;
}

int *crossbowDoubleBufferCapacity (crossbowDoubleBufferP p) {
    nullPointerException (p);
    return p->capacity;
}

void crossbowDoubleBufferRegister (crossbowDoubleBufferP p) {
	int i;
    nullPointerException (p);

    for (i = 0; i < 2; ++i) {

    	/* Register buffers for images */

    	info ("Register buffer #%d for images (%p, %d)\n", i, p->theImages[i], p->capacity[0]);

    	if (mlock (p->theImages[i], p->capacity[0]) < 0)
    		err ("Call to mlock() failed: %s\n", strerror(errno));

    	checkCudaErrors(cudaHostRegister(p->theImages[i], p->capacity[0], cudaHostRegisterMapped | cudaHostRegisterPortable));

    	/* Register buffers for labels */

    	info ("Register buffer #%d for labels\n", i);

    	if (mlock (p->theLabels[i], p->capacity[1]) < 0)
    	    err ("Call to mlock() failed: %s\n", strerror(errno));

    	checkCudaErrors(cudaHostRegister(p->theLabels[i], p->capacity[1], cudaHostRegisterMapped | cudaHostRegisterPortable));
    }

    p->locked = 1;

    return;
}

void crossbowDoubleBufferAdviceWillNeed (crossbowDoubleBufferP p) {
	int i;
	nullPointerException (p);

	for (i = 0; i < 2; ++i) {

		if (madvise(p->theImages[i], p->capacity[0], MADV_WILLNEED | MADV_SEQUENTIAL) != 0) {
				err("Call to madvice() failed: %s\n", strerror(errno));
	    }

		if (madvise(p->theLabels[i], p->capacity[1], MADV_WILLNEED | MADV_SEQUENTIAL) != 0) {
			err("Call to madvice() failed: %s\n", strerror(errno));
		}
	}

	p->needed = 1;
	return;
}

void crossbowDoubleBufferUnregister (crossbowDoubleBufferP p) {

    nullPointerException (p);
    invalidConditionException (p->locked);

    checkCudaErrors(cudaHostUnregister(p->theImages[0]));
    checkCudaErrors(cudaHostUnregister(p->theImages[1]));
    checkCudaErrors(cudaHostUnregister(p->theLabels[0]));
    checkCudaErrors(cudaHostUnregister(p->theLabels[1]));

    p->locked = 0;
    return;
}

void crossbowDoubleBufferLock (crossbowDoubleBufferP p, int ndx) {
	nullPointerException (p);

	unsigned long long spins = 0ULL;

	int offset = DOUBLE_BUFFER_SLOT_SIZE * ndx;
	int comp   = DOUBLE_BUFFER_SLOT_FREE;
	int exch   = DOUBLE_BUFFER_SLOT_BUSY;

	while (comp != __sync_val_compare_and_swap ((int *)(p->slots + offset), comp, exch)) {
		spins ++;
	}

	return;
}

void crossbowDoubleBufferUnlock (crossbowDoubleBufferP p, int ndx) {
	nullPointerException (p);

	unsigned long long spins = 0ULL;

	int offset = DOUBLE_BUFFER_SLOT_SIZE * ndx;
	int comp   = DOUBLE_BUFFER_SLOT_BUSY;
	int exch   = DOUBLE_BUFFER_SLOT_FREE;

	while (comp != __sync_val_compare_and_swap ((int *)(p->slots + offset), comp, exch)) {
		spins ++;
	}

	return;
}

void crossbowDoubleBufferFree (crossbowDoubleBufferP p) {
    if (! p)
        return;

    if (p->locked)
    	crossbowDoubleBufferUnregister (p);

    free (p->theImages[0]);
    free (p->theImages[1]);
    free (p->theLabels[0]);
    free (p->theLabels[1]);

    crossbowFree (p->slots, (DOUBLE_BUFFER_SLOT_SIZE + DOUBLE_BUFFER_SLOT_SIZE));

    crossbowFree (p, sizeof(crossbow_doublebuffer_t));
}

