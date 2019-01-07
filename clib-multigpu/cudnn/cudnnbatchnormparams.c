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

static void __memsetf (crossbowDataBufferP buffer, float value) {
	if (value == 0) {
		checkCudaErrors(cudaMemset (buffer->dev, 0, buffer->size));
	} else {
		int i;
		float *data = (float *) crossbowMalloc (buffer->size);
		int elements = buffer->size / sizeof(float);
		for (i = 0; i < elements; ++i)
			data[i] = value;
		/* Copy data to GPU buffer */
		checkCudaErrors(cudaMemcpy (buffer->dev, (void *) data, buffer->size, cudaMemcpyHostToDevice));
		crossbowFree (data, buffer->size);
	}
	return;
}

/*
 * Assumes that all CUDA calls have been directed to device `ndx`.
 */
void crossbowCudnnBatchNormParamsSetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, int capacity, int master) {
	nullPointerException (p);
	indexOutOfBoundsException (ndx, p->replicas);
	/* Create new buffers for mean and variable and store them in their corresponding array list */
	crossbowDataBufferP m = crossbowDataBufferCreate (capacity, PIN);
	crossbowDataBufferP v = crossbowDataBufferCreate (capacity, PIN);
	/* Initialise them to 0 to 1 respectively */
	__memsetf (m, 0);
	__memsetf (v, 1);
	crossbowArrayListSet (p->mean,     ndx, m);
	crossbowArrayListSet (p->variance, ndx, v);
	/* Create `ready` event to protect specific buffer */
	// checkCudaErrors(cudaEventCreateWithFlags(&(p->ready[ndx]), cudaEventBlockingSync | cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&(p->ready[ndx]), cudaEventDisableTiming));
	/* Reset updates */
	p->updates[ndx] = 0;
	if (master) {
		p->_mean     = crossbowDataBufferCreate (capacity, PIN);
		p->_variance = crossbowDataBufferCreate (capacity, PIN);
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

void crossbowCudnnBatchNormParamsStoreEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, 
	const char *dir, int op) {
	int id;
	crossbowDataBufferP m, v;
	nullPointerException(p);
	for (id = 0; id < p->replicas; ++id) {
		m = (crossbowDataBufferP) crossbowArrayListGet (p->mean,     id);
		v = (crossbowDataBufferP) crossbowArrayListGet (p->variance, id);
		if (! m && ! v)
			continue;
		/* Redirect CUDA calls to corresponding device */
		checkCudaErrors (cudaSetDevice(id));
		char *f = crossbowStringConcat ("%s/gpu-%02d-bn-avg-%03d.dat", dir, id, op);
		char *g = crossbowStringConcat ("%s/gpu-%02d-bn-var-%03d.dat", dir, id, op);
		crossbowDataBufferStore (m, f);
		crossbowDataBufferStore (v, g);
		crossbowStringFree (f);
		crossbowStringFree (g);
	}
	return;
}

void crossbowCudnnBatchNormParamsLoadEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, const char *dir, int op) {
	int id;
	crossbowDataBufferP m, v;
	nullPointerException(p);
	for (id = 0; id < p->replicas; ++id) {
		m = (crossbowDataBufferP) crossbowArrayListGet (p->mean,     id);
		v = (crossbowDataBufferP) crossbowArrayListGet (p->variance, id);
		if (! m && ! v)
			continue;
		/* Redirect CUDA calls to corresponding device */
		checkCudaErrors (cudaSetDevice(id));
		char *f = crossbowStringConcat ("%s/gpu-%02d-bn-avg-%03d.dat", dir, id, op);
		char *g = crossbowStringConcat ("%s/gpu-%02d-bn-var-%03d.dat", dir, id, op);
		crossbowDataBufferLoad (m, f);
		crossbowDataBufferLoad (v, g);
		crossbowStringFree (f);
		crossbowStringFree (g);
	}
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
 */
void crossbowCudnnBatchNormParamsSynchroniseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, crossbowDeviceP defaultDev) {
	int id;
	float one = 1;
	int count = 1;
	float ratio = 1;
	crossbowDataBufferP M, V;
	crossbowDataBufferP m, v;
	
	if (p->replicas == 1)
		return;
	
	M = (crossbowDataBufferP) crossbowArrayListGet (p->mean,     defaultDev->id);
	V = (crossbowDataBufferP) crossbowArrayListGet (p->variance, defaultDev->id);
	
	/* Set device */
	checkCudaErrors (cudaSetDevice(defaultDev->id));
	
	/* Synchronise device */
	checkCudaErrors(cudaDeviceSynchronize());
	
	for (id = 0; id < p->replicas; ++id) {
		if ((id != defaultDev->id) && (p->updates[id] > 0)) {
			
			m = (crossbowDataBufferP) crossbowArrayListGet (p->mean,     id);
			v = (crossbowDataBufferP) crossbowArrayListGet (p->variance, id);
			
			if ((! m) && (! v))
     			continue;

			/* Wait until buffers are ready to use */

			/* Fetch buffers */
			cudaMemcpyPeerAsync (    p->_mean->dev, defaultDev->id, m->dev, id, M->size, defaultDev->modelSynchronisationStream);
			cudaMemcpyPeerAsync (p->_variance->dev, defaultDev->id, v->dev, id, V->size, defaultDev->modelSynchronisationStream);
			
			/* Accumulate buffers */
			checkCublasStatus(cublasSaxpy (defaultDev->modelSynchronisationHandle, M->size / sizeof(float), &(one), (float *)(    p->_mean->dev), 1, (float *)(M->dev), 1));
			checkCublasStatus(cublasSaxpy (defaultDev->modelSynchronisationHandle, V->size / sizeof(float), &(one), (float *)(p->_variance->dev), 1, (float *)(V->dev), 1));
			
			count ++;
		}
	}
	/* Normalise buffers by `count` */
	if (count > 1) {
		ratio = 1. / (float) count;
		checkCublasStatus(cublasSscal(defaultDev->modelSynchronisationHandle, M->size / sizeof(float), &(ratio), (float *)(M->dev), 1));
		checkCublasStatus(cublasSscal(defaultDev->modelSynchronisationHandle, V->size / sizeof(float), &(ratio), (float *)(V->dev), 1));
	}

	/* Copy buffers across devices */
	for (id = 0; id < p->replicas; ++id) {
		if (id != defaultDev->id) {
			m = (crossbowDataBufferP) crossbowArrayListGet (p->mean,     id);
			v = (crossbowDataBufferP) crossbowArrayListGet (p->variance, id);
			/* Copy */
			cudaMemcpyPeerAsync (m->dev, id, M->dev, defaultDev->id, M->size, defaultDev->modelSynchronisationStream);
			cudaMemcpyPeerAsync (v->dev, id, V->dev, defaultDev->id, V->size, defaultDev->modelSynchronisationStream);
		}
	}
	/* Synchronise device */
	checkCudaErrors(cudaDeviceSynchronize());
	
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
		m = crossbowArrayListGet (p->mean,     ndx);
		v = crossbowArrayListGet (p->variance, ndx);
		if (m && v) {
			crossbowDataBufferFree (m);
			crossbowDataBufferFree (v);
			/* Destroy event */
			checkCudaErrors(cudaEventDestroy(p->ready[ndx]));
		}
	}
	/* Free buffer pools */
	crossbowArrayListFree (p->mean);
	crossbowArrayListFree (p->variance);
	/* Free events */
	crossbowFree (p->ready, (p->replicas * sizeof(cudaEvent_t)));
	/* Free updates */
	crossbowFree (p->updates, (p->replicas * sizeof(int)));

	if (p->_mean)
		crossbowDataBufferFree (p->_mean);
	if (p->_variance)
		crossbowDataBufferFree (p->_variance);

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
