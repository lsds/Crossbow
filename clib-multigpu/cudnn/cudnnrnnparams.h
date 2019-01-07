#ifndef __CROSSBOW_CUDNN_RNN_PARAMS_H_
#define __CROSSBOW_CUDNN_RNN_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

typedef struct crossbowCudnnRnnParams *crossbowCudnnRnnParamsP;
typedef struct crossbowCudnnRnnParams {

	crossbowCudnnTensorP input, output;

	cudnnRNNDescriptor_t rnnDescriptor;

	cudnnRNNMode_t computeMode;
	cudnnDirectionMode_t directionMode;
	cudnnRNNInputMode_t inputMode;

	cudnnDropoutDescriptor_t dropoutDescriptor;

} crossbow_cudnn_rnn_params_t;

crossbowCudnnRnnParamsP crossbowCudnnRnnParamsCreate ();

void crossbowCudnnRnnParamsSetInputDescriptor  (crossbowCudnnRnnParamsP);

void crossbowCudnnRnnParamsSetOutputDescriptor  (crossbowCudnnRnnParamsP);

void crossbowCudnnRnnParamsFree (crossbowCudnnRnnParamsP);

char *crossbowCudnnRnnParamsString (crossbowCudnnRnnParamsP);

#endif /* __CROSSBOW_CUDNN_RNN_PARAMS_H_ */
