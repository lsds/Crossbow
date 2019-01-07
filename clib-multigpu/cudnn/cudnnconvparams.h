#ifndef __CROSSBOW_CUDNN_CONV_PARAMS_H_
#define __CROSSBOW_CUDNN_CONV_PARAMS_H_

#include <cudnn.h>

#include "cudnntensor.h"

typedef struct crossbowCudnnConvParams *crossbowCudnnConvParamsP;
typedef struct crossbowCudnnConvParams {

	crossbowCudnnTensorP input, output;

	cudnnFilterDescriptor_t filterDesc;

	int hasBias;
	cudnnTensorDescriptor_t biasDesc;

	cudnnConvolutionDescriptor_t convDesc;

	cudnnConvolutionFwdAlgo_t  forwardAlgorithm;

	cudnnConvolutionBwdFilterAlgo_t backwardFilterAlgorithm;
	cudnnConvolutionBwdDataAlgo_t backwardDataAlgorithm;

	size_t forwardWorkSpaceSize, backwardFilterWorkSpaceSize, backwardDataWorkSpaceSize;

} crossbow_cudnn_conv_params_t;

crossbowCudnnConvParamsP crossbowCudnnConvParamsCreate ();

void crossbowCudnnConvParamsSetInputDescriptor (crossbowCudnnConvParamsP, int, int, int, int);

void crossbowCudnnConvParamsSetOutputDescriptor (crossbowCudnnConvParamsP, int, int, int, int);

void crossbowCudnnConvParamsSetConvolutionDescriptor (crossbowCudnnConvParamsP, int, int, int, int);

void crossbowCudnnConvParamsSetFilterDescriptor (crossbowCudnnConvParamsP, int, int, int, int);

void crossbowCudnnConvParamsSetBiasDescriptor (crossbowCudnnConvParamsP, int, int, int, int);

size_t crossbowCudnnConvParamsConfigureForward (crossbowCudnnConvParamsP, int, double, cudnnHandle_t);

size_t crossbowCudnnConvParamsConfigureBackwardFilter (crossbowCudnnConvParamsP, int, double, cudnnHandle_t);

size_t crossbowCudnnConvParamsConfigureBackwardData (crossbowCudnnConvParamsP, int, double, cudnnHandle_t);

char *crossbowCudnnConvParamsString (crossbowCudnnConvParamsP);

void crossbowCudnnConvParamsFree (crossbowCudnnConvParamsP);

#endif /* __CROSSBOW_CUDNN_CONV_PARAMS_H_ */
