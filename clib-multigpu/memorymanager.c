#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "waitfreequeue.h"
#include "timer.h"

#include <sys/mman.h>
#include <errno.h>
#include <string.h>

#include <sched.h>

/*
 * Manage/track calls to:
 *
 * a) malloc (incl. string copies)
 * b) cudaMallocHost
 * c) cudaMalloc
 *
 * a) free
 * b) cudaFreeHost
 * c) cudaFree
 */

#ifndef INC_ATOMICALLY
#define INC_ATOMICALLY
#endif

static long mallocCalls = 0L;
static long cudaMallocHostCalls = 0L;
static long cudaMallocCalls = 0L;

static long freeCalls = 0L;
static long cudaFreeHostCalls = 0L;
static long cudaFreeCalls = 0L;

static long mallocBytes = 0L;
static long cudaMallocHostBytes = 0L;
static long cudaMallocBytes = 0L;

static int pagesize;

void *crossbowMalloc (int size) {
	void *p;
	p = malloc (size);
	if (! p) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
#ifndef INC_ATOMICALLY
	mallocBytes += size;
	mallocCalls ++;
#else
	/* Atomically increment counter */
	__sync_add_and_fetch (&(mallocBytes), size);
	__sync_add_and_fetch (&(mallocCalls),    1);
#endif
	return p;
}

void *crossbowMallocAligned (int alignment, int size) {
	void *p;
	p = aligned_alloc (alignment, size);
	if (! p) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
#ifndef INC_ATOMICALLY
	mallocBytes += size;
	mallocCalls ++;
#else
	/* Atomically increment counter */
	__sync_add_and_fetch (&(mallocBytes), size);
	__sync_add_and_fetch (&(mallocCalls),    1);
#endif
	return p;
}

void *crossbowCudaMallocHost (int size) {
	void *p = NULL;
	checkCudaErrors(cudaMallocHost (&p, size));
	/* checkCudaErrors(cudaHostAlloc (&p, size, cudaHostAllocPortable)); */
#ifndef INC_ATOMICALLY
	cudaMallocHostBytes += size;
	cudaMallocHostCalls ++;
#else
	__sync_add_and_fetch (&(cudaMallocHostBytes), size);
	__sync_add_and_fetch (&(cudaMallocHostCalls),    1);
#endif
	return p;
}

void *crossbowCudaMalloc (int size) {
	void *p = NULL;
	checkCudaErrors(cudaMalloc (&p, size));
#ifndef INC_ATOMICALLY
	cudaMallocBytes += size;
	cudaMallocCalls ++;
#else
	__sync_add_and_fetch (&(cudaMallocBytes), size);
	__sync_add_and_fetch (&(cudaMallocCalls),    1);
#endif
	return p;
}

void *crossbowFree (void *item, int size) {
	if (item) {
		free (item);
#ifndef INC_ATOMICALLY
		mallocBytes -= size;
		freeCalls ++;
#else
		__sync_sub_and_fetch (&(mallocBytes), size);
		__sync_add_and_fetch (&(freeCalls),      1);
#endif
	}
	return NULL;
}

void *crossbowCudaFreeHost (void *item, int size) {
	if (item) {
		checkCudaErrors(cudaFreeHost (item));
#ifndef INC_ATOMICALLY
		cudaMallocHostBytes -= size;
		cudaFreeHostCalls ++;
#else
		__sync_sub_and_fetch (&(cudaMallocHostBytes), size);
		__sync_add_and_fetch (&(cudaFreeHostCalls),      1);
#endif
	}
	return NULL;
}

void *crossbowCudaFree (void *item, int size) {
	if (item) {
		checkCudaErrors(cudaFree (item));
#ifndef INC_ATOMICALLY
		cudaMallocBytes -= size;
		cudaFreeCalls ++;
#else
		__sync_sub_and_fetch (&(cudaMallocBytes), size);
		__sync_add_and_fetch (&(cudaFreeCalls),      1);
#endif
	}
	return NULL;
}

char *crossbowStringCopy (const char *s) {
	size_t len = strlen(s) + 1;
	void *t = crossbowMalloc (len);
	return (char *) memcpy (t, s, len);
}

int crossbowStringAppend (char *s, int *offset, size_t *remain, const char * format, ...) {
	int r;
	int p, n;
	va_list args;
	p = *offset;
	n = *remain;
	va_start (args, format);
	r = vsnprintf(s + p, n, format, args);
	if (r < 0 || r > n) {
		fprintf(stderr, "error: string buffer overflow\n");
		exit (1);
	}
	va_end (args);
	*offset += r;
	*remain -= r;
	return r;
}

char *crossbowStringConcat (const char * format, ...) {
	int r;
	va_list args;
	char s [1024];
	int n = sizeof(s) - 1;
	memset (s, 0, sizeof(s));
	va_start (args, format);
	r = vsnprintf (s, n, format, args);
	if (r < 0 || r > n) {
		fprintf(stderr, "error: string buffer overflow\n");
		exit (1);
	}
	va_end (args);
	return crossbowStringCopy (s);
}

void *crossbowStringFree (const char *s) {
	size_t len = strlen (s) + 1;
	crossbowFree ((void *) s, len);
	return NULL;
}

void crossbowMemoryManagerInit () {

	pagesize = getpagesize();
	return;
}

void crossbowMemoryManagerDestroy () {
	return;
}

static inline int __is_pointer_aligned (const void *p, int alignment) {
    return ((((uintptr_t) p) & (alignment - 1)) == 0);
}

static inline int __is_length_aligned (int length, int alignment) {
    return ((length & (alignment - 1)) == 0);
}

static inline unsigned isRegistered (void *buffer, int start, int end) {
	/*
	(void) buffer;
	(void) start;
	(void) end;
	*/
	
	struct cudaPointerAttributes attributes;

	void *p = (void *) ((char *) (buffer) + start);
	void *q = (void *) ((char *) (buffer) +   end);

	while (p != q) {
		if (cudaPointerGetAttributes (&attributes, p) != cudaSuccess) {
			cudaGetLastError ();
			return 0;
		}
		p = q; /* (void *) ((char *) (p) + pagesize); */
	}
	
	return 1;
}

void crossbowHostRegisterBuffer (int type, void *buffer, int length, int start, int end, crossbowPhase_t phase) {
	
	/* Check if buffer pointer is page-aligned */
	if (! __is_pointer_aligned (buffer, pagesize))
		err("Buffer pointer %p (%s %s) is not page-aligned\n", buffer, (phase == 0) ? "training" : "test", (type == 0) ? "examples" : "labels");

	/* Check if buffer length is page-aligned */
	if (! __is_length_aligned (length, pagesize))
		err("Buffer length %d (%s %s) is not page-aligned\n", length, (phase == 0) ? "training" : "test", (type == 0) ? "examples" : "labels");

	/* Check if task length is page-aligned */
	if (! __is_length_aligned ((end - start), pagesize))
		err("Task length %d (%s %s) is not page-aligned\n", (end - start), (phase == 0) ? "training" : "test", (type == 0) ? "examples" : "labels");

	while (! isRegistered (buffer, start, end)) {
		/* Calling this method as an alternative to Thread.yield (via JNI) */
		sched_yield ();
	}

	return;
}

void crossbowMemoryManagerDump () {
	size_t free, total;
	cudaMemGetInfo(&free, &total);

	printf ("=== [Memory manager] ===\n");

	printf("%ld bytes of virtual memory (%ld malloc's %ld free's)\n",
			mallocBytes, mallocCalls, freeCalls);

	printf("%ld bytes pinned (%ld malloc's %ld free's)\n",
			cudaMallocHostBytes, cudaMallocHostCalls, cudaFreeHostCalls);

	printf("%ld bytes on device memory (%ld malloc's %ld free's) %zu/%zu bytes remaining\n",
			cudaMallocBytes, cudaMallocCalls, cudaFreeCalls, free, total);

	printf ("=== [End of memory manager dump] ===\n");
	fflush (stdout);
	return;
}
