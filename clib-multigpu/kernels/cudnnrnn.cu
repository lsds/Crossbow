#include "cudnnrnn.h"

void crossbowCudnnKernelRnn (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	crossbowCudnnRnnParamsP params = s->op->kernel->descriptors.rnn;
	
	if (s->phi == CHECK) {
		
	} else {
		
	}

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
