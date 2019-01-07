#include "cudnndropoutgradient.h"


void crossbowCudnnKernelDropoutGradient (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get cuDNN dropout parameters */
	crossbowCudnnDropoutParamsP params = s->op->peer->kernel->descriptors.dropout;
#ifdef GPU_VERBOSE
	char *str = crossbowCudnnDropoutParamsString (params);
	info ("%s\n", str);
	crossbowStringFree (str);
#endif

	/* Get workspace (from peer) */
	crossbowDataBufferP reserveSpace = NULL;
	if (! crossbowListEmpty (s->locals[s->op->peer->id]))
		reserveSpace = (crossbowDataBufferP) crossbowListPeek (s->locals[s->op->peer->id], 0);

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnDropoutBackward(s->cudnnHandle[s->op->branch],
			params->dropoutDesc[s->deviceId],
			params->input->descriptor,
			input->dev,
			params->output->descriptor,
			output->dev,
			(reserveSpace != NULL) ? reserveSpace->dev : NULL,
			params->reserveSpaceSize
			));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (params);
	UNUSED (input);
	UNUSED (output);
	UNUSED (reserveSpace);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
