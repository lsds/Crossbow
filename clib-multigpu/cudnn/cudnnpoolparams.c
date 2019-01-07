#include "cudnnpoolparams.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "cudnnhelper.h"

crossbowCudnnPoolParamsP crossbowCudnnPoolParamsCreate () {

	crossbowCudnnPoolParamsP p;

	p = (crossbowCudnnPoolParamsP) crossbowMalloc (sizeof(crossbow_cudnn_pool_params_t));

	// dbg("Pooling parameters @%p\n", p);

	memset (p, 0, sizeof(crossbow_cudnn_pool_params_t));

	return p;
}

void crossbowCudnnPoolParamsSetInputDescriptor (crossbowCudnnPoolParamsP p, int count, int channels, int height, int width) {

	p->input = crossbowCudnnTensorCreate (count, channels, height, width);
}

void crossbowCudnnPoolParamsSetOutputDescriptor (crossbowCudnnPoolParamsP p, int count, int channels, int height, int width) {

	p->output = crossbowCudnnTensorCreate (count, channels, height, width);
}

void crossbowCudnnPoolParamsSetMode (crossbowCudnnPoolParamsP p, int mode) {
	switch (mode) {
	case 0: p->mode = CUDNN_POOLING_MAX;
	        break;
	case 1: p->mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	        break;
	default:
		err("invalid cuDNN pooling mode");
	}
}

void crossbowCudnnPoolParamsSetPoolingDescriptor (crossbowCudnnPoolParamsP p, int windowHeight, int windowWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth) {

	checkCudnnStatus(cudnnCreatePoolingDescriptor(&(p->poolDesc)));
	/* Initialise it into a 2D description */

	checkCudnnStatus(cudnnSetPooling2dDescriptor(p->poolDesc, p->mode, CUDNN_PROPAGATE_NAN, windowHeight, windowWidth, paddingHeight, paddingWidth, strideHeight, strideWidth));
}

char *crossbowCudnnPoolParamsString (crossbowCudnnPoolParamsP p) {

	cudnnPoolingMode_t mode;
	cudnnNanPropagation_t maxpoolingNanOpt;
	int windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride;

	char s [1024];
	int offset;
	size_t remaining;

	checkCudnnStatus(cudnnGetPooling2dDescriptor (p->poolDesc, &mode, &maxpoolingNanOpt, &windowHeight, &windowWidth, &verticalPadding, &horizontalPadding, &verticalStride, &horizontalStride));

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	/* Get tensor descriptors */
	char *d1 = crossbowCudnnTensorString (p->input);
	char *d2 = crossbowCudnnTensorString (p->output);

	/* input [n, c, h, w] output [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "input %s output %s (mode %s, %s, window [%d, %d] padding [%d, %d], stride [%d, %d])", d1, d2, cudnnPoolingModeString(mode), cudnnNanPropagationString(maxpoolingNanOpt), windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

	crossbowStringFree (d1);
	crossbowStringFree (d2);

	return crossbowStringCopy (s);
}

void crossbowCudnnPoolParamsFree (crossbowCudnnPoolParamsP p) {

	if (! p)
		return;

	crossbowCudnnTensorFree (p->input);
	crossbowCudnnTensorFree (p->output);

	checkCudnnStatus(cudnnDestroyPoolingDescriptor(p->poolDesc));

	crossbowFree (p, sizeof(crossbow_cudnn_pool_params_t));
}
