#include "cudnnpoolgradient.h"

void crossbowCudnnKernelPoolGradient (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer and reset it */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Fill output buffer with zeros */
#ifndef CUDART_NOOP
	checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (output);
#endif

	nullPointerException(s->op->peer);

	/* Get input of peer operator */
	crossbowDataBufferP peerInput = crossbowStreamGetPeerInput (s);

	/* Get output of peer operator */
	crossbowDataBufferP peerOutput = crossbowStreamGetPeerOutput (s);

	/* Get cuDNN pooling parameters (from peer operator) */
	crossbowCudnnPoolParamsP params = s->op->peer->kernel->descriptors.pool;

	float alpha = 1;
	float beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnPoolingBackward(s->cudnnHandle[s->op->branch],
			params->poolDesc,
			&alpha,
			params->output->descriptor,
			peerOutput->dev,
			params->output->descriptor,
			input->dev,
			params->input->descriptor,
			peerInput->dev,
			&beta,
			params->input->descriptor,
			output->dev
			));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (params);
	UNUSED (alpha);
	UNUSED (peerOutput);
	UNUSED (input);
	UNUSED (peerInput);
	UNUSED (beta);
	UNUSED (output);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
