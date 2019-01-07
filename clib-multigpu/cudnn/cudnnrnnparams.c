#include "cudnnbatchnormparams.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "cudnnhelper.h"

crossbowCudnnBatchNormParamsP crossbowCudnnBatchNormParamsCreate (int replicas) {
	crossbowCudnnBatchNormParamsP p;
	p = (crossbowCudnnBatchNormParamsP) crossbowMalloc (sizeof(crossbow_cudnn_batchnorm_params_t));
	/* Initialise to 0 */
	memset (p, 0, sizeof(crossbow_cudnn_batchnorm_params_t));
	/* Initialise state */
	p->replicas = replicas;
	p->mean = crossbowArrayListCreate (p->replicas);
	p->variance = crossbowArrayListCreate (p->replicas);
	p->ready = (cudaEvent_t *) crossbowMalloc (p->replicas * sizeof(cudaEvent_t));
	memset (p->ready, 0, (p->replicas * sizeof(cudaEvent_t)));
	p->updates = (int *) crossbowMalloc (p->replicas * sizeof(int));
	memset (p->updates, 0, (p->replicas * sizeof(int)));
	p->_mean = NULL;
	p->_variance = NULL;
	return p;
}

void crossbowCudnnBatchNormParamsSetInputDescriptor (crossbowCudnnBatchNormParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	dbg("Set batchnorm filter descriptor [%d, %d, %d, %d]\n", count, channels, height, width);
	p->input = crossbowCudnnTensorCreate (count, channels, height, width);
	return;
}

void crossbowCudnnBatchNormParamsSetOutputDescriptor (crossbowCudnnBatchNormParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	dbg("Set batchnorm filter descriptor [%d, %d, %d, %d]\n", count, channels, height, width);
	p->output = crossbowCudnnTensorCreate (count, channels, height, width);
	return;
}

void crossbowCudnnBatchNormParamsSetBatchNormDescriptor (crossbowCudnnBatchNormParamsP p) {
	nullPointerException(p);
	checkCudnnStatus (cudnnCreateTensorDescriptor(&(p->derivedBatchNormDesc)));
	checkCudnnStatus (cudnnDeriveBNTensorDescriptor(p->derivedBatchNormDesc, p->input->descriptor, CUDNN_BATCHNORM_SPATIAL));
	return;
}

/*
 * Assumes that all CUDA calls have been directed to device `ndx`.
 */
void crossbowCudnnBatchNormParamsSetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, int capacity, int master) {
	nullPointerException (p);
	indexOutOfBoundsException (ndx, p->replicas);
	/* Create new buffers for mean and variable and store them in their corresponding array list */
	crossbowDataBufferP m = crossbowDataBufferCreate (capacity, REF);
	crossbowDataBufferP v = crossbowDataBufferCreate (capacity, REF);
	crossbowArrayListSet (p->mean,     ndx, m);
	crossbowArrayListSet (p->variance, ndx, v);
	/* Create `ready` event to protect specific buffer */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->ready[ndx]), cudaEventBlockingSync | cudaEventDisableTiming));
	/* Reset updates */
	p->updates[ndx] = 0;
	if (master) {
		p->_mean     = crossbowDataBufferCreate (capacity, REF);
		p->_variance = crossbowDataBufferCreate (capacity, REF);
	}
	return;
}

void crossbowCudnnBatchNormParamsGetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, cudaStream_t stream, crossbowDataBufferP *m, crossbowDataBufferP *v, int *f) {
	nullPointerException(p);
	indexOutOfBoundsException (ndx, p->replicas);
	/* Wait for ready event */
	checkCudaErrors(cudaStreamWaitEvent(stream, p->ready[ndx], 0));
	*m = (crossbowDataBufferP) crossbowArrayListGet (p->mean,     ndx);
	*v = (crossbowDataBufferP) crossbowArrayListGet (p->variance, ndx);
	*f = p->updates[ndx];
	return;
}

void crossbowCudnnBatchNormParamsReleaseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, cudaStream_t stream) {
	nullPointerException(p);
	indexOutOfBoundsException (ndx, p->replicas);
	/* Record event */
	checkCudaErrors(cudaEventRecord (p->ready[ndx], stream));
	/* Increment updates */
	p->updates[ndx] ++;
}

/*
 * Average the mean and variance buffers across devices.
 *
 * We assume that `s` is the default device's model synchronisation stream;
 * and that all CUDA calls have been directed to the default device.
 */
void crossbowCudnnBatchNormParamsSynchroniseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, crossbowDeviceP defaultDev) {
	int id;
	int count = 0;
	for (id = 0; id < p->replicas; ++id) {
		if ((id != defaultDev->id) && (p->updates[id] > 0)) {

			/* Wait until buffers are ready to use */

			/* Fetch buffers */

			/* Accumulate buffers */

			count ++;
		}
	}
	/* Normalise buffers by `count` */

	/* Copy buffers across devices */
	for (id = 0; id < p->replicas; ++id) {
		if (id != defaultDev->id) {
			/* Copy */
		}
	}
	/* Reset updates per device */
	memset (p->updates, 0, (p->replicas * sizeof(int)));
	return;
}

void crossbowCudnnBatchNormParamsFree (crossbowCudnnBatchNormParamsP p) {
	int ndx;
	crossbowDataBufferP m, v;

	if (! p)
		return;

	crossbowCudnnTensorFree (p->input);
	crossbowCudnnTensorFree (p->output);

	/* Destroy batch norm descriptor */

	/* Free buffers & events */
	for (ndx = 0; ndx < p->replicas; ndx++) {
		m = crossbowArrayListGet (p->mean, ndx);
		v = crossbowArrayListGet (p->mean, ndx);
		if (m && v) {
			crossbowDataBufferFree (m);
			crossbowDataBufferFree (v);
			/* Destroy event */

		}
	}
	/* Free buffer pools */
	crossbowArrayListFree (p->mean);
	crossbowArrayListFree (p->variance);
	/* Free events */
	crossbowFree (p->ready, (p->replicas * sizeof(cudaEvent_t)));
	/* Free updates */
	crossbowFree (p->updates, (p->replicas * sizeof(int)));

	crossbowFree (p, sizeof(crossbow_cudnn_batchnorm_params_t));
	return;
}

char *crossbowCudnnBatchNormParamsString (crossbowCudnnBatchNormParamsP p) {

	cudnnDataType_t type;
	int n, c, h, w;
	int nStride, cStride, hStride, wStride;

	char s [1024];
	int offset;
	size_t remaining;

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	/* Get tensor descriptors */
	char *d1 = crossbowCudnnTensorString (p->input);
	char *d2 = crossbowCudnnTensorString (p->output);

	/* input [n, c, h, w] output [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "input %s output %s", d1, d2);

	checkCudnnStatus(cudnnGetTensor4dDescriptor (p->derivedBatchNormDesc, &type, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));

	crossbowStringAppend (s, &offset, &remaining, " descriptor [%d, %d, %d, %d]", n, c, h, w);

	crossbowStringFree (d1);
	crossbowStringFree (d2);

	return crossbowStringCopy (s);
}
