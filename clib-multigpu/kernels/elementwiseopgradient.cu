#include "elementwiseopgradient.h"

void crossbowKernelElementWiseOpGradient (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Number of elements in input variable */
	int elements = s->op->kernel->inputs[0]->schema->elements;

	/* Find number of downstream operators */
	/* int numberOfPeerInputs = crossbowArrayListSize (s->op->peer->upstream); */
	int numberOfPeerInputs = 1;

	/* Get kernel configuration parameter */
	int numberOfCoefficients = 0;
	float *coefficients = crossbowKernelConfigParamGetFloatArray 
        ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0), &numberOfCoefficients);
	/* Currently, only sum is supported */
	for (int i = 0; i < numberOfPeerInputs; ++i) {
		/* Get an output buffer */
		crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);
		/* Fill output buffer with zeros */
#ifndef CUDART_NOOP
        /* This is required because of the saxpy operation. */
		checkCudaErrors (cudaMemsetAsync (
            output->dev, 
            0, 
            s->op->kernel->output->schema->bytes, 
            s->stream[s->op->branch]));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (output);
#endif
        /* saxpy using coefficient */
#ifndef CUBLAS_NOOP
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(coefficients[i]), (float *) (input->dev), 1, (float *) (output->dev), 1));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (elements);
        UNUSED (coefficients);
        UNUSED (input);
        UNUSED (output);
#endif
        /* Append output to list */
		crossbowListAppend (s->outputs[s->op->id], output);
	}

	return;
}
