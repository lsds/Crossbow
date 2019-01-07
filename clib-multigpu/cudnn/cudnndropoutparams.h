#ifndef __CROSSBOW_CUDNN_DROPOUT_PARAMS_H_
#define __CROSSBOW_CUDNN_DROPOUT_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

#include "../arraylist.h"

typedef struct crossbowCudnnDropoutParams *crossbowCudnnDropoutParamsP;
typedef struct crossbowCudnnDropoutParams {

	crossbowCudnnTensorP input, output;

	int replicas;
	cudnnDropoutDescriptor_t *dropoutDesc;
	crossbowArrayListP states;

	size_t reserveSpaceSize;

} crossbow_cudnn_dropout_params_t;

crossbowCudnnDropoutParamsP crossbowCudnnDropoutParamsCreate (int);

void crossbowCudnnDropoutParamsSetInputDescriptor (crossbowCudnnDropoutParamsP, int, int, int, int);

void crossbowCudnnDropoutParamsSetOutputDescriptor (crossbowCudnnDropoutParamsP, int, int, int, int);

void crossbowCudnnDropoutParamsSetDropoutDescriptor (crossbowCudnnDropoutParamsP, int, cudnnHandle_t, float, unsigned long long);

size_t crossbowCudnnDropoutParamsGetReserveSpaceSize (crossbowCudnnDropoutParamsP);

char *crossbowCudnnDropoutParamsString (crossbowCudnnDropoutParamsP);

void crossbowCudnnDropoutParamsFree (crossbowCudnnDropoutParamsP);

#endif /* __CROSSBOW_CUDNN_DROPOUT_PARAMS_H_ */
