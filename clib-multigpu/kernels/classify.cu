#include "classify.h"

__global__ void crossbowKernelClassifyCompute (
	const int elements,
	const int numberofclasses,
	const float* probabilities,
	float* output) {

	CUDA_KERNEL_LOOP(index, elements) {

		/* Move pointer to the beginning of values for specific element */
		const float *p = &probabilities[index * numberofclasses];

		float maxvalue = FLT_MIN;
		int actuallabel = -1;

		for (int i = 0; i < numberofclasses; ++i) {
			if (maxvalue < p[i]) {
				maxvalue = p[i];
				actuallabel = i;
			}
		}

		output[index] = (float) (actuallabel);
	}
}

void crossbowKernelClassify (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input variable */
	crossbowVariableP theInput = (crossbowVariableP) s->op->kernel->inputs[0];
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Elements should be equal to the batch size */
	int elements = crossbowVariableSchemaShape(theInput->schema, 0);
	int numberofclasses = theInput->schema->elements / elements;

#ifndef KERNEL_NOOP
	crossbowKernelClassifyCompute<<<GET_BLOCKS(elements), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>
		(
			elements,
			numberofclasses,
			(float *) (input->dev), /* SoftMax probabilities */
			(float *) (output->dev)
		);
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (elements);
	UNUSED (numberofclasses);
	UNUSED (input);
	UNUSED (output);
#endif
	
	/* Store output */
	crossbowListAppend (s->outputs[s->op->id], output);
	
	return;
}
