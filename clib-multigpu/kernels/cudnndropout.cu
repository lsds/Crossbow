#include "cudnndropout.h"


void crossbowCudnnKernelDropout (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;
	
	crossbowVariableP theInput = (crossbowVariableP) (s->op->kernel->inputs[0]);
    int length = theInput->schema->bytes;
	
	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get cuDNN dropout parameters */
	crossbowCudnnDropoutParamsP params = s->op->kernel->descriptors.dropout;
#ifdef GPU_VERBOSE
	char *str = crossbowCudnnDropoutParamsString (params);
	info ("%s\n", str);
	crossbowStringFree (str);
#endif

	/* Get workspace */
	int reserveSpaceSizeInBytes = 0;
	crossbowDataBufferP reserveSpace = NULL;

	if (params->reserveSpaceSize > 0) {

		reserveSpace = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, NULL, &reserveSpaceSizeInBytes);
		nullPointerException(reserveSpace);
		dbg("Got reserve space %p of size %d bytes\n", reserveSpace, reserveSpaceSizeInBytes);
	}
	
	if (s->phi == CHECK) {
		/* Simply copy input to output (dropout should not be called during inference) */
#ifndef CUDART_NOOP
		cudaMemcpyAsync (output->dev, input->dev, length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
#else
		/* Subterfuge unused parameter warnings */
		UNUSED (output);
		UNUSED (input);
		UNUSED (length);
#endif
	} else {
#ifndef CUDANN_NOOP
		checkCudnnStatus(cudnnDropoutForward(s->cudnnHandle[s->op->branch],
			params->dropoutDesc[s->deviceId],
			params->input->descriptor,
			input->dev,
			params->output->descriptor,
			output->dev,
			(reserveSpace != NULL) ? reserveSpace->dev : NULL,
			reserveSpaceSizeInBytes /* Must equal params->reserveSpaceSize */
			));
#else
		/* Subterfuge unused parameter warnings */
		UNUSED (params);
		UNUSED (input);
		UNUSED (output);
		UNUSED (reserveSpace);
		UNUSED (reserveSpaceSizeInBytes);
#endif
	}

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	/* Return read-write local variables to kernel when the dataflow execution completes */
	if (reserveSpace)
		crossbowListAppend (s->locals[s->op->id], reserveSpace);

	return;
}

