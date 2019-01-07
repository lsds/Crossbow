#include "cudnnrelugradient.h"

void crossbowCudnnKernelReLUGradient (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get kernel configuration parameters */
	float slope = crossbowKernelConfigParamGetFloatValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));

	if (slope != 0)
		err ("error: slope must be 0");
	/*
	 * FIXME
	 *
	 * Variable `slope` is not used locally - it only the check that it's 0
	 * that interests us, so we could elevate the latter to the Java code.
	 */

	nullPointerException(s->op->peer);

	/* Get input of peer operator */
	crossbowDataBufferP peerInput = crossbowStreamGetPeerInput (s);

	/* Get output of peer operator */
	crossbowDataBufferP peerOutput = crossbowStreamGetPeerOutput (s);

	/* Get cuDNN pooling parameters */
	crossbowCudnnReLUParamsP params = s->op->peer->kernel->descriptors.relu;

	float alpha = 1;
	float beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnActivationBackward(s->cudnnHandle[s->op->branch],
			params->activationDesc,
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
