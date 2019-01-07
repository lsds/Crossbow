#include "datasetfile.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "memoryregionpool.h"

crossbowDatasetFileP crossbowDatasetFileCreate (const char *filename) {
	crossbowDatasetFileP p;
	p = (crossbowDatasetFileP) crossbowMalloc(sizeof(crossbow_datasetfile_t));
	memset (p, 0, sizeof(crossbow_datasetfile_t));
	p->filename = crossbowStringCopy (filename);
	p->opened = 0;
	p->mapped = 0;
	p->locked = 0;
	p->needed = 0;
	p->region = NULL;
	crossbowDatasetFileOpen (p);
#ifndef __LAZY_MAPPING
	crossbowDatasetFileMap (p);
#endif
	return p;
}

void crossbowDatasetFileOpen (crossbowDatasetFileP p) {
	nullPointerException (p);
	if (p->opened)
		return;
	p->fd = open(p->filename, O_RDWR);
	if (p->fd < 0) {
		fprintf(stderr, "error: failed to open %s\n", p->filename);
		exit (1);
	}
	crossbowDatasetFileStat (p);
	p->opened = 1;
}

void crossbowDatasetFileStat (crossbowDatasetFileP p) {
	struct stat sb;
	if (fstat(p->fd, &sb) < 0) {
		fprintf(stderr, "error: failed to stat %s\n", p->filename);
		exit (1);
	}
	p->length = (int) sb.st_size;
}

void crossbowDatasetFileMap (crossbowDatasetFileP p) {
	nullPointerException (p);
	if (p->mapped)
		return;
	p->data = mmap(0, p->length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_NORESERVE, p->fd, 0);
	if (p->data == MAP_FAILED) {
		fprintf(stderr, "error: failed to map %s\n", p->filename);
		exit (1);
	}
	p->mapped = 1;
}

void crossbowDatasetFileAssign (crossbowDatasetFileP p, void *region) {
	nullPointerException (p);
    invalidConditionException (p->region == NULL);
	p->region = region;
	return;
}

void crossbowDatasetFileWithdraw (crossbowDatasetFileP p, void *region) {
	nullPointerException (p);
    invalidConditionException (p->region == region);
	p->region = NULL;
	return;
}

/**
 * This call should be considered thread-safe
 */
void crossbowDatasetFileRegister (crossbowDatasetFileP p, int blocksize) {
	unsigned i;
	unsigned blocks;
	void *ptr;
	/* Make sure that the file has not been registered before */
	invalidConditionException(p->locked == 0);
	invalidArgumentException ((p->length % blocksize) == 0);
	blocks = p->length / blocksize;
	ptr = p->data;
	for (i = 0; i < blocks; ++i) {
		if (mlock (ptr, blocksize) < 0)
			err ("Call to mlock() failed: %s\n", strerror(errno));
		checkCudaErrors(cudaHostRegister(ptr, blocksize, cudaHostRegisterMapped | cudaHostRegisterPortable));
		ptr = (void *) ((char *) (ptr) + (blocksize));
	}
	p->locked = blocks;
	return;
}

void crossbowDatasetFileRegisterRegion (crossbowDatasetFileP p, int offset, int length) {
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (mlock (ptr, length) < 0)
		err ("Call to mlock() failed: %s\n", strerror(errno));
    /* dbg("Register %s: %p offset %d length %d\n", p->filename, ptr, offset, length); */
	checkCudaErrors(cudaHostRegister(ptr, length, cudaHostRegisterMapped | cudaHostRegisterPortable));
	/* Atomically increment counter */
	__sync_add_and_fetch (&(p->locked), 1);
	return;
}

void crossbowDatasetFileAdviceWillNeed (crossbowDatasetFileP p) {
	invalidConditionException(p->mapped);
	if (madvise(p->data, p->length, MADV_WILLNEED | MADV_SEQUENTIAL) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	p->needed = 1;
	return;
}

void crossbowDatasetFileAdviceWillNeedRegion (crossbowDatasetFileP p, int offset, int length) {
	invalidConditionException(p->mapped);
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (madvise(ptr, length, MADV_WILLNEED | MADV_SEQUENTIAL) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	/* Atomically increment counter */
	__sync_add_and_fetch (&(p->needed), 1);
	return;
}

int crossbowDatasetFileSize (crossbowDatasetFileP p) {
	nullPointerException (p);
	invalidConditionException(p->opened);
	return p->length;
}

unsigned long crossbowDatasetFileAddress (crossbowDatasetFileP p) {
	nullPointerException (p);
	/* invalidConditionException(p->mapped); */

	if (! p->mapped)
		return (unsigned long) 0;

	return (unsigned long) (p->data);
}

void crossbowDatasetFileAdviceDontNeed (crossbowDatasetFileP p) {
	invalidConditionException(p->mapped);
	if (madvise(p->data, p->length, MADV_DONTNEED) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	p->needed = 0;
	return;
}

void crossbowDatasetFileAdviceDontNeedRegion (crossbowDatasetFileP p, int offset, int length) {
#ifdef __LAZY_MAPPING
	int pending;
#endif
	invalidConditionException(p->mapped);
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (madvise(ptr, length, MADV_DONTNEED) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
#ifdef __LAZY_MAPPING
	pending = __sync_sub_and_fetch (&(p->needed), 1);
	if (! pending) {
		/* At this point, all regions have been unlocked and unregistered */
		crossbowDatasetFileUnmap (p);
	}
#else
	__sync_sub_and_fetch (&(p->needed), 1);
#endif
	return;
}

void crossbowDatasetFileUnregister (crossbowDatasetFileP p, int blocksize) {
	unsigned i;
	unsigned blocks;
	void *ptr;
	invalidArgumentException ((p->length % blocksize) == 0);
	blocks = p->length / blocksize;
	/* Assert that all blocks have been registered */
	invalidConditionException(p->locked == blocks);
	ptr = p->data;
	dbg("Unregister %s: %p %d blocks\n", p->filename, ptr, blocks);
	for (i = 0; i < blocks; ++i) {
        dbg("Unregister block #%03d\n", i);
		checkCudaErrors(cudaHostUnregister(ptr));
		if (munlock (ptr, blocksize) < 0)
			err("Call to munlock() failed: %s\n", strerror(errno));
		ptr = (void *) ((char *) (ptr) + (blocksize));
	}
	/* Reset counter */
	p->locked = 0;
	return;
}

void crossbowDatasetFileUnregisterRegion (crossbowDatasetFileP p, int offset, int length) {
	void *ptr = (void *) ((char *) (p->data) + offset);
	if (munlock (ptr, length) < 0)
		err ("Call to munlock() failed: %s\n", strerror(errno));
	checkCudaErrors(cudaHostUnregister(ptr));
	/* Atomically increment counter */
	__sync_sub_and_fetch (&(p->locked), 1);
	return;
}

void crossbowDatasetFileUnmap (crossbowDatasetFileP p) {
	nullPointerException (p);
	if (! p->mapped)
		return;
	munmap (p->data, p->length);
	p->mapped = 0;
}

void crossbowDatasetFileClose (crossbowDatasetFileP p) {
	if (! p->opened)
		return;
	close (p->fd);
	p->opened = 0;
	return;
}

void crossbowDatasetFileFree (crossbowDatasetFileP p, int block) {
	if (! p)
		return;
	if (p->locked)
		crossbowDatasetFileUnregister (p, block);
	if (p->region) {
		/* Return memory region back to its pool */
		crossbowMemoryRegionP region = (crossbowMemoryRegionP) p->region;
		info("Memory region #%03d currently in use; returning it\n", region->id);
		crossbowMemoryRegionPoolRelease ((crossbowMemoryRegionPoolP) region->pool, region);
	}
	crossbowDatasetFileUnmap (p);
	crossbowDatasetFileClose (p);
	crossbowStringFree (p->filename);
	crossbowFree(p, sizeof(crossbow_datasetfile_t));
}
