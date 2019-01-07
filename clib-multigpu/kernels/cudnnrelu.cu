#include "cudnnrelu.h"

void crossbowCudnnKernelReLU (void *args) {

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
	 * Variable `slope` is not used locally - it is only the check that it's 0
	 * that interests us, so we could elevate the latter to the Java code.
	 */

	/* Get cuDNN pooling parameters */
	crossbowCudnnReLUParamsP params = s->op->kernel->descriptors.relu;

	float alpha = 1;
	float beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnActivationForward(s->cudnnHandle[s->op->branch],
			params->activationDesc,
			&alpha,
			params->input->descriptor,
			input->dev,
			&beta,
			params->output->descriptor,
			output->dev
			));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (params);
	UNUSED (alpha);
	UNUSED (input);
	UNUSED (beta);
	UNUSED (output);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
