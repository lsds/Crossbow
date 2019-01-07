#ifndef __CROSSBOW_DEVICE_H_
#define __CROSSBOW_DEVICE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#include <cudnn.h>

#include <curand.h>

#include "utils.h"

typedef struct crossbow_device *crossbowDeviceP;
typedef struct crossbow_device {
	int id;
	unsigned selected;

	int64_t frequency;

	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;

	curandGenerator_t curandGenerator;

	cublasHandle_t modelSynchronisationHandle;
	cudaStream_t   modelSynchronisationStream;

#ifdef MAKESPAN_MEASUREMENTS
	cudaEvent_t barrier;
#endif

} crossbow_device_t;

crossbowDeviceP crossbowDeviceCreate (int);

void crossbowDeviceSelect (crossbowDeviceP);

unsigned crossbowDeviceSelected (crossbowDeviceP);

void crossbowDeviceFree (crossbowDeviceP);

#endif /* __CROSSBOW_DEVICE_H_ */
