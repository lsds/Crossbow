#include "cudnnreluparams.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "cudnnhelper.h"

crossbowCudnnReLUParamsP crossbowCudnnReLUParamsCreate () {

	crossbowCudnnReLUParamsP p;

	p = (crossbowCudnnReLUParamsP) crossbowMalloc (sizeof(crossbow_cudnn_relu_params_t));
	
	/* Initialise to 0 */
	memset (p, 0, sizeof(crossbow_cudnn_relu_params_t));
	return p;
}

void crossbowCudnnReLUParamsSetActivationDescriptor (crossbowCudnnReLUParamsP p, int type, double ceiling) {
	nullPointerException(p);
	/*
	 * cudnnActivationMode_t is an enum to select the neuron activation function
	 * used in cudnnActivationForward() and cudnnActivationBackward().
	 *
	 * Values are:
	 *
	 * CUDNN_ACTIVATION_SIGMOID
	 * CUDNN_ACTIVATION_RELU
	 * CUDNN_ACTIVATION_TANH (hyperbolic tangent)
	 * CUDNN_ACTIVATION_CLIPPED_RELU
	 *
	 * Note: CUDA 6.0 supports ELU also.
	 */
	checkCudnnStatus(cudnnCreateActivationDescriptor(&(p->activationDesc)));
	cudnnActivationMode_t mode;
	switch (type) {
	case 0: mode = CUDNN_ACTIVATION_SIGMOID;      break;
	case 1: mode = CUDNN_ACTIVATION_RELU;         break;
	case 2: mode = CUDNN_ACTIVATION_TANH;         break;
	case 3: mode = CUDNN_ACTIVATION_CLIPPED_RELU; break;
	default:
		err("Invalid activation mode %d\n", type);
	}
	/* Initialise it into a 2D description */
	checkCudnnStatus(cudnnSetActivationDescriptor(p->activationDesc, mode, CUDNN_PROPAGATE_NAN, ceiling));
	return;
}

void crossbowCudnnReLUParamsSetInputDescriptor (crossbowCudnnReLUParamsP p, int count, int channels, int height, int width) {

	p->input = crossbowCudnnTensorCreate (count, channels, height, width);
}

void crossbowCudnnReLUParamsSetOutputDescriptor (crossbowCudnnReLUParamsP p, int count, int channels, int height, int width) {

	p->output = crossbowCudnnTensorCreate (count, channels, height, width);
}

char *crossbowCudnnReLUParamsString (crossbowCudnnReLUParamsP p) {

	cudnnActivationMode_t mode;
	cudnnNanPropagation_t reluNanOpt;
	double ceil;

	char s [1024];
	int offset;
	size_t remaining;

	checkCudnnStatus(cudnnGetActivationDescriptor (p->activationDesc, &mode, &reluNanOpt, &ceil));

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	/* Get tensor descriptors */
	char *d1 = crossbowCudnnTensorString (p->input);
	char *d2 = crossbowCudnnTensorString (p->output);

	/* input [n, c, h, w] output [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "input %s output %s (mode %s, %s, ceil %.1f)", d1, d2, cudnnActivationModeString(mode), cudnnNanPropagationString(reluNanOpt), ceil);

	crossbowStringFree (d1);
	crossbowStringFree (d2);

	return crossbowStringCopy (s);
}

void crossbowCudnnReLUParamsFree (crossbowCudnnReLUParamsP p) {

	if (! p)
		return;

	crossbowCudnnTensorFree (p->input);
	crossbowCudnnTensorFree (p->output);

	checkCudnnStatus(cudnnDestroyActivationDescriptor(p->activationDesc));

	crossbowFree (p, sizeof(crossbow_cudnn_relu_params_t));
}
