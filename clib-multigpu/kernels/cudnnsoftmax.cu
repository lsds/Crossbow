#include "cudnnsoftmax.h"

void crossbowCudnnKernelSoftMax (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get cuDNN pooling parameters */
	crossbowCudnnSoftMaxParamsP params = s->op->kernel->descriptors.softmax;

	float alpha = 1;
	float beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnSoftmaxForward(s->cudnnHandle[s->op->branch],
			CUDNN_SOFTMAX_ACCURATE,
		    CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			params->input->descriptor,
			input->dev,
			&beta,
			params->output->descriptor,
			output->dev
			));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (alpha);
    UNUSED (params);
    UNUSED (input);
    UNUSED (beta);
    UNUSED (output);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
