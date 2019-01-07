#include "cudnnsoftmaxgradient.h"

void crossbowCudnnKernelSoftMaxGradient (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer and reset it */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);
	/* Fill output buffer with zeros */
#ifndef CUDART_NOOP
    /* checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch])); */
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (output);
#endif

	nullPointerException(s->op->peer);

	/* Get output of peer operator */
	crossbowDataBufferP peerOutput = crossbowStreamGetPeerOutput (s);

	/* Get cuDNN pooling parameters */
	crossbowCudnnSoftMaxParamsP params = s->op->peer->kernel->descriptors.softmax;

	float alpha = 1;
	float beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnSoftmaxBackward(s->cudnnHandle[s->op->branch],
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			params->output->descriptor,
			peerOutput->dev,
			params->output->descriptor,
			input->dev,
			&beta,
			params->input->descriptor,
			output->dev
			));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (alpha);
    UNUSED (params);
    UNUSED (peerOutput);
    UNUSED (input);
    UNUSED (beta);
    UNUSED (output);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
