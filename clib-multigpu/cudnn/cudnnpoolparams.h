#ifndef __CROSSBOW_CUDNN_POOL_PARAMS_H_
#define __CROSSBOW_CUDNN_POOL_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

typedef struct crossbowCudnnPoolParams *crossbowCudnnPoolParamsP;
typedef struct crossbowCudnnPoolParams {

	crossbowCudnnTensorP input, output;

	cudnnPoolingMode_t mode;
	cudnnPoolingDescriptor_t poolDesc;

} crossbow_cudnn_pool_params_t;

crossbowCudnnPoolParamsP crossbowCudnnPoolParamsCreate ();

void crossbowCudnnPoolParamsSetInputDescriptor (crossbowCudnnPoolParamsP, int, int, int, int);

void crossbowCudnnPoolParamsSetOutputDescriptor (crossbowCudnnPoolParamsP, int, int, int, int);

void crossbowCudnnPoolParamsSetMode (crossbowCudnnPoolParamsP, int);

void crossbowCudnnPoolParamsSetPoolingDescriptor (crossbowCudnnPoolParamsP, int, int, int, int, int, int);

char *crossbowCudnnPoolParamsString (crossbowCudnnPoolParamsP);

void crossbowCudnnPoolParamsFree (crossbowCudnnPoolParamsP);

#endif /* __CROSSBOW_CUDNN_POOL_PARAMS_H_ */
