 #include "elementwiseop.h"

void crossbowKernelElementWiseOp (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get output buffer */
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
	
    /* Number of elements in output variable (they should match elements per input) */
	int elements = s->op->kernel->output->schema->elements;

	/* Find number of upstream operators */
	int numberOfInputs = crossbowArrayListSize (s->op->upstream);

	/* Get kernel configuration parameter */
	int numberOfCoefficients = 0;
	float *coefficients = crossbowKernelConfigParamGetFloatArray 
        ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0), &numberOfCoefficients);
	invalidConditionException (numberOfCoefficients == numberOfInputs);

	/* Currently, only sum is supported */

	for (int i = 0; i < numberOfInputs; ++i) {

		crossbowOperatorP prev = (crossbowOperatorP) crossbowArrayListGet (s->op->upstream, i);
		dbg("Element-wise operator input #%d is from operator %d (kernel %s)\n", i, prev->id, prev->kernel->name);
		/* Get output of previous operator */
		crossbowDataBufferP input = crossbowStreamOperatorGetOutput (s, prev);
		/* saxpy using coefficient */
#ifndef CUBLAS_NOOP
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(coefficients[i]), (float *) (input->dev), 1, (float *) (output->dev), 1));
#else
        UNUSED (elements);
        UNUSED (coefficients);
        UNUSED (input);
        UNUSED (output);
#endif
    }

	crossbowListAppend (s->outputs[s->op->id], output);

	return;
}
