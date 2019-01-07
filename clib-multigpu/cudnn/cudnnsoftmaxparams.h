#ifndef __CROSSBOW_CUDNN_SOFTMAX_PARAMS_H_
#define __CROSSBOW_CUDNN_SOFTMAX_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

typedef struct crossbowCudnnSoftMaxParams *crossbowCudnnSoftMaxParamsP;
typedef struct crossbowCudnnSoftMaxParams {

	crossbowCudnnTensorP input, output;

} crossbow_cudnn_softmax_params_t;

crossbowCudnnSoftMaxParamsP crossbowCudnnSoftMaxParamsCreate ();

void crossbowCudnnSoftMaxParamsSetInputDescriptor (crossbowCudnnSoftMaxParamsP, int, int, int, int);

void crossbowCudnnSoftMaxParamsSetOutputDescriptor (crossbowCudnnSoftMaxParamsP, int, int, int, int);

char *crossbowCudnnSoftMaxParamsString (crossbowCudnnSoftMaxParamsP);

void crossbowCudnnSoftMaxParamsFree (crossbowCudnnSoftMaxParamsP);

#endif /* __CROSSBOW_CUDNN_SOFTMAX_PARAMS_H_ */
