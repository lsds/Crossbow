#ifndef __CROSSBOW_CUDNN_BATCHNORM_PARAMS_H_
#define __CROSSBOW_CUDNN_BATCHNORM_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

#include "../arraylist.h"
#include "../databuffer.h"
#include "../device.h"

typedef struct crossbowCudnnBatchNormParams *crossbowCudnnBatchNormParamsP;
typedef struct crossbowCudnnBatchNormParams {

	crossbowCudnnTensorP input, output;

	cudnnTensorDescriptor_t derivedBatchNormDesc;

	int replicas;
	crossbowArrayListP mean;
	crossbowArrayListP variance;
	/* "Mutex" to protect access to mean and variance buffers (per device)
	 *
	 * When an event is recorded, the buffer is ready to be reused.
	 * A thread waits for an event to be recorded before using that
	 * buffer.
	 */
	cudaEvent_t *ready;
	/* Number of times mean and variance buffers have been updated (per device) */
	int *updates;
	/* Buffers allocated only on the master device for synchronisation purposes */
	crossbowDataBufferP _mean, _variance;

} crossbow_cudnn_batchnorm_params_t;

crossbowCudnnBatchNormParamsP crossbowCudnnBatchNormParamsCreate (int);

void crossbowCudnnBatchNormParamsSetInputDescriptor  (crossbowCudnnBatchNormParamsP, int, int, int, int);

void crossbowCudnnBatchNormParamsSetOutputDescriptor (crossbowCudnnBatchNormParamsP, int, int, int, int);

void crossbowCudnnBatchNormParamsSetBatchNormDescriptor (crossbowCudnnBatchNormParamsP);

void crossbowCudnnBatchNormParamsSetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, int, int, int);

void crossbowCudnnBatchNormParamsGetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, int, cudaStream_t, crossbowDataBufferP *, crossbowDataBufferP *, int *);

void crossbowCudnnBatchNormParamsReleaseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, int, cudaStream_t);

void crossbowCudnnBatchNormParamsSynchroniseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, crossbowDeviceP);

void crossbowCudnnBatchNormParamsStoreEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, const char *, int);

void crossbowCudnnBatchNormParamsLoadEstimatedMeanAndVariable  (crossbowCudnnBatchNormParamsP, const char *, int);

void crossbowCudnnBatchNormParamsFree ();

char *crossbowCudnnBatchNormParamsString (crossbowCudnnBatchNormParamsP);

#endif /* __CROSSBOW_CUDNN_BATCHNORM_PARAMS_H_ */
