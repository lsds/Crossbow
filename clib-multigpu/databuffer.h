#ifndef __CROSSBOW_DATABUFFER_H_
#define __CROSSBOW_DATABUFFER_H_

#include "thetaqueue.h"
#include "utils.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

/* Data buffer parts may live only on the device */
typedef enum crossbow_databuffer_location { host, device, bilocated } crossbow_databuffer_location_t;

typedef struct crossbow_databuffer *crossbowDataBufferP;
typedef struct crossbow_databuffer {
	int size;
	crossbowDataBuffer_t type;
	void *host;
	void *dev;
	crossbowThetaQueueP queue;
	int index;
	unsigned refs; /* Reference count */
	unsigned shifted;
	void *host_;
	void *dev_;
} crossbow_databuffer_t;

crossbowDataBufferP crossbowDataBufferCreate (int size, crossbowDataBuffer_t);

crossbowDataBufferP crossbowDataBufferReplicate (crossbowDataBufferP);

void *crossbowDataBufferGetHostPointer (crossbowDataBufferP);

void crossbowDataBufferSetHostPointer (crossbowDataBufferP, void *);

void *crossbowDataBufferGetDevicePointer (crossbowDataBufferP);

void crossbowDataBufferInitHostRegion (crossbowDataBufferP, int, void *, int, int);

void crossbowDataBufferCopyDeviceRegion (crossbowDataBufferP, crossbowDataBufferP, cudaStream_t);

int crossbowDataBufferSize (crossbowDataBufferP);

void crossbowDataBufferPush (crossbowDataBufferP, cudaStream_t);

void crossbowDataBufferPushSync (crossbowDataBufferP);

void crossbowDataBufferPushRegion (crossbowDataBufferP, void *, int, int, cudaStream_t);

void crossbowDataBufferPull (crossbowDataBufferP, cudaStream_t);

void crossbowDataBufferPullSync (crossbowDataBufferP);

void crossbowDataBufferRelease (crossbowDataBufferP);

float crossbowDataBufferComputeCheckSum (crossbowDataBufferP, int, int);

int crossbowDataBufferComputeCheckSumAsInt (crossbowDataBufferP, int, int);

void crossbowDataBufferWriteToFile (crossbowDataBufferP, const char *, int, int, int);

int crossbowDataBufferStore (crossbowDataBufferP, const char *);
int crossbowDataBufferLoad  (crossbowDataBufferP, const char *);

void crossbowDataBufferShift (crossbowDataBufferP, int bytes);
void crossbowDataBufferReset (crossbowDataBufferP);

void crossbowDataBufferFree (crossbowDataBufferP);

#endif /* __CROSSBOW_DATABUFFER_H_ */
