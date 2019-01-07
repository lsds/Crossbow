#include "noopstateless.h"

void crossbowKernelNoopStateless (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	crossbowVariableP theInput = (crossbowVariableP) (s->op->kernel->inputs[0]);

	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);
	int length = theInput->schema->bytes;

	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

#ifndef CUDART_NOOP
	cudaMemcpyAsync (output->dev, input->dev, length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
#else
	UNUSED (output);
	UNUSED (input);
	UNUSED (length);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
