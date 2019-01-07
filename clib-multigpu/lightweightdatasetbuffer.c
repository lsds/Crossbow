#include "lightweightdatasetbuffer.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <errno.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/mman.h> /* mlock, etc. */
#include <unistd.h>

crossbowLightWeightDatasetBufferP crossbowLightWeightDatasetBufferCreate (int capacity) {
	crossbowLightWeightDatasetBufferP p = NULL;
	int error;
	p = (crossbowLightWeightDatasetBufferP) crossbowMalloc (sizeof(crossbow_lightweightdatasetbuffer_t));
	p->capacity = capacity;
	error = posix_memalign(&(p->data), getpagesize(), p->capacity);
	if (error) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
	/* Counters */
	p->locked = 0;
	p->needed = 0;
	return p;
}

unsigned long crossbowLightWeightDatasetBufferAddress (crossbowLightWeightDatasetBufferP p) {
	nullPointerException (p);
	return (unsigned long) (p->data);
}

int crossbowLightWeightDatasetBufferCapacity (crossbowLightWeightDatasetBufferP p) {
	nullPointerException (p);
	return p->capacity;
}

void crossbowLightWeightDatasetBufferRegister (crossbowLightWeightDatasetBufferP p, int blocksize) {
	unsigned i;
	unsigned blocks;
	void *ptr;
	/* Make sure that the file has not been registered before */
	invalidConditionException(p->locked == 0);
	invalidArgumentException ((p->capacity % blocksize) == 0);
	blocks = p->capacity / blocksize;
	ptr = p->data;
	info("Register %p (%d blocks)\n", ptr, blocks);
	for (i = 0; i < blocks; ++i) {
		if (mlock (ptr, blocksize) < 0)
			err ("Call to mlock() failed: %s\n", strerror(errno));
		checkCudaErrors(cudaHostRegister(ptr, blocksize, cudaHostRegisterMapped | cudaHostRegisterPortable));
		ptr = (void *) ((char *) (ptr) + (blocksize));
	}
	p->locked = blocks;
	return;
}

void crossbowLightWeightDatasetBufferAdviceWillNeed (crossbowLightWeightDatasetBufferP p) {
	nullPointerException (p);
	if (madvise(p->data, p->capacity, MADV_WILLNEED | MADV_SEQUENTIAL) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	p->needed = 1;
	return;
}

void crossbowLightWeightDatasetBufferUnregister (crossbowLightWeightDatasetBufferP p, int blocksize) {
	unsigned i;
	unsigned blocks;
	void *ptr;
	invalidArgumentException ((p->capacity % blocksize) == 0);
	blocks = p->capacity / blocksize;
	/* Assert that all blocks have been registered */
	invalidConditionException(p->locked == blocks);
	ptr = p->data;
	info("Unregister %p (%d blocks)\n", ptr, blocks);
	for (i = 0; i < blocks; ++i) {
		if (munlock (ptr, blocksize) < 0)
			err("Call to munlock() failed: %s\n", strerror(errno));
		checkCudaErrors(cudaHostUnregister(ptr));
		ptr = (void *) ((char *) (ptr) + (blocksize));
	}
	/* Reset counter */
	p->locked = 0;
	return;
}

void crossbowLightWeightDatasetBufferAdviceDontNeed (crossbowLightWeightDatasetBufferP p) {
	nullPointerException (p);
	if (madvise(p->data, p->capacity, MADV_DONTNEED) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	p->needed = 0;
	return;
}

void crossbowLightWeightDatasetBufferCopy (crossbowLightWeightDatasetBufferP p, int thisoffset, void *data, int dataoffset, int bytes) {
	nullPointerException (p);
	nullPointerException (data);
	void *dst = (void *) ((char *) (p->data) + thisoffset);
	void *src = (void *) ((char *) (   data) + dataoffset);
	memcpy (dst, src, bytes);
	return;
}

void crossbowLightWeightDatasetBufferFree (crossbowLightWeightDatasetBufferP p, int blocksize) {
	if (! p)
		return;
	if (p->locked)
		crossbowLightWeightDatasetBufferUnregister (p, blocksize);
	free (p->data);
	crossbowFree (p, sizeof(crossbow_lightweightdatasetbuffer_t));
}

