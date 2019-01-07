#include "memoryregion.h"

#include "memoryregionpool.h" /* Used to release memory region */

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <unistd.h>

crossbowMemoryRegionP crossbowMemoryRegionCreate (int id, int capacity, void *pool) {
	crossbowMemoryRegionP p = NULL;
	int error;
	p = (crossbowMemoryRegionP) crossbowMalloc (sizeof(crossbow_memory_region_t));
	p->id = id;
    p->capacity = capacity;
	p->limit = 0;
	error = posix_memalign(&(p->data), getpagesize(), p->capacity);
	if (error) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
	p->next = NULL;
	p->file = NULL;
    p->pool = pool;
	/* Counters */
	p->locked = 0;
	p->needed = 0;
    return p;
}

void crossbowMemoryRegionReset (crossbowMemoryRegionP p) {
	nullPointerException(p);
	p->limit = 0;
	p->file = NULL;
	p->locked = 0;
	p->needed = 0;
	return;
}

void crossbowMemoryRegionSetLimit (crossbowMemoryRegionP p, int limit) {
	nullPointerException(p);
    invalidConditionException (p->limit == 0);
    invalidConditionException (p->capacity >= limit);
	p->limit = limit;
	return;
}

void crossbowMemoryRegionCopyBlock (crossbowMemoryRegionP p, int offset, int length, int pad) {
	nullPointerException (p);
	/* info("Memory region %p limit %10d offset %10d length %10d\n", p->data, p->limit, offset, length); */
    invalidConditionException ((p->limit > 0) && (p->limit >= (offset + length)));
    /*
     * Assumptions:
     *
     *  i) Offset is a multiple of length (length equals task size with padding)
     * ii) Therefore, (offset / length) returns the block id.
     */
    invalidConditionException ((offset % length) ==  0);   
	int fileoffset = offset - ((offset / length) * pad);

	invalidConditionException ((fileoffset + length - pad) <=  p->file->length);
    
	void *dst = (void *) ((char *) (p->data)       +     offset);
	void *src = (void *) ((char *) (p->file->data) + fileoffset);
	/* Reset memory region block */
	memset (dst, 0, length);
    /* Be careful not to copy a few extra bytes */
	memcpy (dst, src, (length - pad));
	return;
}

unsigned long crossbowMemoryRegionAddress (crossbowMemoryRegionP p) {
	return (unsigned long) (p->data);
}

void crossbowMemoryRegionRegister (crossbowMemoryRegionP p, int offset, int length) {
	nullPointerException(p);
    invalidConditionException ((p->limit > 0) && (p->limit >= (offset + length)));
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (mlock (ptr, length) < 0)
		err ("Call to mlock() failed: %s\n", strerror(errno));
	checkCudaErrors(cudaHostRegister(ptr, length, cudaHostRegisterMapped | cudaHostRegisterPortable));
	/* Atomically increment counter */
	__sync_add_and_fetch (&(p->locked), 1);
	return;
}

void crossbowMemoryRegionAdviceWillNeed (crossbowMemoryRegionP p, int offset, int length) {
	nullPointerException(p);
    invalidConditionException ((p->limit > 0) && (p->limit >= (offset + length)));
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (madvise(ptr, length, MADV_WILLNEED | MADV_SEQUENTIAL) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	/* Atomically increment counter */
	__sync_add_and_fetch (&(p->needed), 1);
	return;
}

void crossbowMemoryRegionUnregister (crossbowMemoryRegionP p, int offset, int length) {
	nullPointerException(p);
    invalidConditionException ((p->limit > 0) && (p->limit >= (offset + length)));
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (munlock (ptr, length) < 0)
		err ("Call to munlock() failed: %s\n", strerror(errno));
	checkCudaErrors(cudaHostUnregister(ptr));
	/* Atomically increment counter */
	__sync_sub_and_fetch (&(p->locked), 1);
	return;
}

void crossbowMemoryRegionAdviceDontNeed (crossbowMemoryRegionP p, int offset, int length) {
#ifdef __LAZY_MAPPING
	int pending;
#endif
	nullPointerException(p);
    invalidConditionException ((p->limit > 0) && (p->limit >= (offset + length)));
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (madvise(ptr, length, MADV_DONTNEED) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
#ifdef __LAZY_MAPPING
	pending = __sync_sub_and_fetch (&(p->needed), 1);
    /* 
     * Thread-safe code, assuming that the next time the file
     * slides in again will be in a while.
     */
	if (! pending) {
		/* 
         * At this point, all regions have been unlocked and unregistered:
         *
         * a) Unmap the data set file
         * b) Decouple the file from this memory region
         * c) Return the memory region to the pool
         */
		crossbowDatasetFileUnmap    (p->file);
        crossbowDatasetFileWithdraw (p->file, p);
        
		crossbowMemoryRegionPoolRelease ((crossbowMemoryRegionPoolP) p->pool, p);
	}
#else
	__sync_sub_and_fetch (&(p->needed), 1);
#endif
	return;
}

void crossbowMemoryRegionFree (crossbowMemoryRegionP p) {
	if (! p)
		return;
	free (p->data);
	crossbowFree (p, sizeof(crossbow_memory_region_t));
}
