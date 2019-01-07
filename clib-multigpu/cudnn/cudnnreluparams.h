#ifndef __CROSSBOW_CUDNN_RELU_PARAMS_H_
#define __CROSSBOW_CUDNN_RELU_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

typedef struct crossbowCudnnReLUParams *crossbowCudnnReLUParamsP;
typedef struct crossbowCudnnReLUParams {

	crossbowCudnnTensorP input, output;

	cudnnActivationDescriptor_t activationDesc;

} crossbow_cudnn_relu_params_t;

crossbowCudnnReLUParamsP crossbowCudnnReLUParamsCreate ();

void crossbowCudnnReLUParamsSetInputDescriptor (crossbowCudnnReLUParamsP, int, int, int, int);

void crossbowCudnnReLUParamsSetOutputDescriptor (crossbowCudnnReLUParamsP, int, int, int, int);

void crossbowCudnnReLUParamsSetActivationDescriptor (crossbowCudnnReLUParamsP, int, double);

char *crossbowCudnnReLUParamsString (crossbowCudnnReLUParamsP);

void crossbowCudnnReLUParamsFree (crossbowCudnnReLUParamsP);

#endif /* __CROSSBOW_CUDNN_RELU_PARAMS_H_ */
