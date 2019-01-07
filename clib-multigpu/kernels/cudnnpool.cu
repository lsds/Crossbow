#include "cudnnpool.h"

void crossbowCudnnKernelPool (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get cuDNN pooling parameters */
	crossbowCudnnPoolParamsP params = s->op->kernel->descriptors.pool;
#ifdef GPU_VERBOSE
	char *str = crossbowCudnnPoolParamsString (params);
	info ("%s\n", str);
	crossbowStringFree (str);
#endif

	float alpha = 1;
	float beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnPoolingForward(s->cudnnHandle[s->op->branch],
			params->poolDesc,
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

	/* Pull output for debugging */
	/*
	crossbowDataBufferPullSync (output);

	float *top = (float *) (output->host);
	float sum = 0;
	for (int i = 0; i < s->op->kernel->output->schema->elements; i++) {
		sum += top[i];
	}
	printf("[DBG] pool output sum (%d elements) = %15.5f\n", s->op->kernel->output->schema->elements, sum);
	*/

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
