#include "databuffer.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

crossbowDataBufferP crossbowDataBufferCreate (int size, crossbowDataBuffer_t type) {
	crossbowDataBufferP p;
	p = (crossbowDataBufferP) crossbowMalloc (sizeof(crossbow_databuffer_t));
	p->size = size;
	p->type = type;
	if (p->type == PIN) {
		p->host = crossbowCudaMallocHost (p->size);
		memset(p->host, 0, p->size);
	} else {
		p->host = NULL;
	}
	/* Always allocate buffer on device memory */
	p->dev = crossbowCudaMalloc (p->size);

	p->queue = NULL;
	p->index = 0;

	p->refs = 0;
	/* New logic to handle multiple tasks per input batch */
	p->shifted = 0;
	p->host_ = p->host;
	p->dev_  = p->dev;
	return p;
}

crossbowDataBufferP crossbowDataBufferReplicate (crossbowDataBufferP buffer) {
	crossbowDataBufferP p;
	p = (crossbowDataBufferP) crossbowMalloc (sizeof(crossbow_databuffer_t));
	p->size = buffer->size;
	p->type = buffer->type;
	if (p->type == PIN) {
		p->host = crossbowCudaMallocHost (p->size);
		memcpy (p->host, buffer->host, p->size);
	} else {
		p->host = NULL;
	}
	p->dev = crossbowCudaMalloc (p->size);
	/* Copy GPU device memory region as well; use default stream */
	checkCudaErrors(cudaMemcpy (p->dev, buffer->dev, p->size, cudaMemcpyDeviceToDevice));

	p->queue = NULL;
	p->index = 0;

	p->refs = 0;
	return p;
}

void *crossbowDataBufferGetHostPointer (crossbowDataBufferP p) {
	nullPointerException(p->host);
	return p->host;
}

void crossbowDataBufferSetHostPointer (crossbowDataBufferP p, void *data) {
	if (p->type == PIN)
		illegalOperationException();
	p->host = data;
	return;
}

void *crossbowDataBufferGetDevicePointer (crossbowDataBufferP p) {
	return p->dev;
}

int crossbowDataBufferSize (crossbowDataBufferP p) {
	return p->size;
}

void crossbowDataBufferInitHostRegion (crossbowDataBufferP p, int offset, void *source, int start, int length) {
	if (p->type != PIN)
		illegalOperationException();
	memcpy ((void *)(((char *) p->host) + offset), (void *)(((char *) source) + start), length);
}

void crossbowDataBufferCopyDeviceRegion (crossbowDataBufferP p, crossbowDataBufferP q, cudaStream_t stream) {
	/* Copy q's GPU device buffer to p's GPU device buffer */
	nullPointerException(p);
	nullPointerException(q);
	invalidConditionException(p->size == q->size);
	checkCudaErrors(cudaMemcpyAsync (p->dev, q->dev, p->size, cudaMemcpyDeviceToDevice, stream));
}

/* Push `host` to GPU device */
void crossbowDataBufferPush (crossbowDataBufferP p, cudaStream_t stream) {
	checkCudaErrors(cudaMemcpyAsync (p->dev, p->host, p->size, cudaMemcpyHostToDevice, stream));
	return;
}

void crossbowDataBufferPushSync (crossbowDataBufferP p) {
	checkCudaErrors(cudaMemcpy (p->dev, p->host, p->size, cudaMemcpyHostToDevice));
	return;
}

/* Push `data` to GPU device */
void crossbowDataBufferPushRegion (crossbowDataBufferP p, void *data, int offset, int length, cudaStream_t stream) {
	checkCudaErrors(cudaMemcpyAsync (((char *) (p->dev) + offset), data, length, cudaMemcpyHostToDevice, stream));
	return;
}

/* Pull data from GPU device */
void crossbowDataBufferPull (crossbowDataBufferP p, cudaStream_t stream) {
	nullPointerException (p->host);
	checkCudaErrors(cudaMemcpyAsync (p->host, p->dev, p->size, cudaMemcpyDeviceToHost, stream));
	return;
}

void crossbowDataBufferPullSync (crossbowDataBufferP p) {
	nullPointerException (p->host);
	checkCudaErrors(cudaMemcpy (p->host, p->dev, p->size, cudaMemcpyDeviceToHost));
	return;
}

void crossbowDataBufferRelease (crossbowDataBufferP p) {
	crossbowThetaQueueP queue = p->queue;
	int index = p->index;

	p->queue = NULL;
	p->index = 0;
	/*
	 * When p->queue is null, the buffer is never reused neither is freed (leading to a memory leak).
	 * It was set to null only for debugging purposes.
	 *
	 * nullPointerException (p->queue);
	 */
	if (! queue)
		return;
	crossbowThetaQueueRelease (queue, index);
}

float crossbowDataBufferComputeCheckSum (crossbowDataBufferP p, int offset, int bytes) {

	int i;
	float checksum = 0;

	nullPointerException(p);

	void *t = (void *) crossbowMalloc (bytes);

	checkCudaErrors(cudaMemcpy (t, (void *)(((char *) p->dev) + offset), bytes, cudaMemcpyDeviceToHost));

	float *values = (float *) (t);

	invalidConditionException((bytes % sizeof(float)) == 0);
	int elements = bytes / sizeof(float);

	for (i = 0; i < elements; ++i)
		checksum += values[i];

	crossbowFree (t, bytes);

	return checksum;
}

int crossbowDataBufferComputeCheckSumAsInt (crossbowDataBufferP p, int offset, int bytes) {

	int i;
	int checksum = 0;

	nullPointerException(p);

	void *t = (void *) crossbowMalloc (bytes);

	checkCudaErrors(cudaMemcpy (t, (void *)(((char *) p->dev) + offset), bytes, cudaMemcpyDeviceToHost));

	int *values = (int *) (t);

	invalidConditionException((bytes % sizeof(int)) == 0);
	int elements = bytes / sizeof(int);

	for (i = 0; i < elements; ++i)
		checksum += values[i];

	crossbowFree (t, bytes);

	return checksum;
}

void crossbowDataBufferWriteToFile (crossbowDataBufferP p, const char *prefix, int task, int offset, int bytes) {
	char *filename = crossbowStringConcat ("%s-%d.txt", prefix, task);
	/* Create new file */
	FILE *fp = fopen(filename, "w");
	if (fp == NULL)
		err ("Failed to open %s\n", filename);

	void *t = (void *) crossbowMalloc (bytes);
	checkCudaErrors(cudaMemcpy (t, (void *)(((char *) p->dev) + offset), bytes, cudaMemcpyDeviceToHost));
	
	float *values = (float *) (t);

	invalidConditionException((bytes % sizeof(float)) == 0);
	int elements = bytes / sizeof(float);
	
	int i;
	float v;
	for (i = 0; i < elements; ++i) {
		v = values[i];
		fprintf(fp, "%.10f\n", v);
	}
	
	crossbowFree (t, bytes);
    crossbowStringFree (filename);
}

int crossbowDataBufferStore (crossbowDataBufferP p, const char *filename) {
#ifdef GPU_VERBOSE
	/* Debugging */
	float checksum = crossbowDataBufferComputeCheckSum (p, 0, p->size);
	dbg ("Store data buffer %p with checksum %.5f\n", p, checksum);
#endif
	void *t = NULL;
	t = (p->type == PIN) ? p->host : (void *) crossbowMalloc (p->size);
	/* Fetch GPU data to host memory */
	checkCudaErrors(cudaMemcpy (t, p->dev, p->size, cudaMemcpyDeviceToHost));
	/* Create new file */
	int fd = open (filename, O_WRONLY | O_CREAT, 0644);
	if (fd < 0)
		err ("Failed to open %s\n", filename);
	/* Write host buffer to file */
	int bytes = (int) write(fd, t, p->size);
	if (bytes != p->size)
		err("%d/%d bytes written to %s\n", bytes, p->size, filename);
	close(fd);
	if (p->type != PIN)
		crossbowFree (t, p->size);
	return 0;
}

int crossbowDataBufferLoad (crossbowDataBufferP p, const char *filename) {
	void *t = NULL;
	t = (p->type == PIN) ? p->host : (void *) crossbowMalloc (p->size);
	int fd = open (filename, O_RDONLY);
	if (fd < 0)
		err ("Failed to open %s\n", filename);
	/* Read buffer from file */
	int bytes = (int) read(fd, t, p->size);
	if (bytes != p->size)
		err("%d/%d bytes read from %s\n", bytes, p->size, filename);
	/* Copy host buffer to GPU memory */
	checkCudaErrors(cudaMemcpy (p->dev, t, p->size, cudaMemcpyHostToDevice));
	close(fd);
	if (p->type != PIN)
		crossbowFree (t, p->size);
#ifdef GPU_VERBOSE
	float checksum = crossbowDataBufferComputeCheckSum (p, 0, p->size);
	dbg ("Loaded data buffer %p with checksum %.5f\n", p, checksum);
#endif
	return 0;
}

void crossbowDataBufferShift (crossbowDataBufferP p, int bytes) {
	nullPointerException (p);
	/* Check that new pointer does not exceed original pointer */

	p->shifted += 1;

	if (p->host)
		p->host = (void *)(((char *) p->host) + bytes);

	p->dev = (void *)(((char *) p->dev) + bytes);
	return;
}

void crossbowDataBufferReset (crossbowDataBufferP p) {
	nullPointerException (p);
	if (p->shifted) {
		p->shifted = 0;
		p->host = p->host_;
		p->dev  = p->dev_;
	}
	return;
}

void crossbowDataBufferFree (crossbowDataBufferP p) {
	if (! p)
		return;
	if (p->type == PIN)
		crossbowCudaFreeHost (p->host, p->size);
	crossbowCudaFree (p->dev, p->size);
	crossbowFree (p, sizeof(crossbow_databuffer_t));
	return;
}
