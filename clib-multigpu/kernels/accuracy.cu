#include "accuracy.h"

__global__ void crossbowKernelAccuracyCompute (
	const int elements,
	const int numberofclasses,
	const float* probabilities,
	const int* labels,
	float* count) {

	CUDA_KERNEL_LOOP(index, elements) {

		int expectedlabel = labels [index];

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

		/* Normalise value to [0,1] */
		count[index] = (actuallabel == expectedlabel) ? (1. / (float) elements) : 0;
	}
}

void crossbowKernelAccuracy (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input variable */
	crossbowVariableP theInput = (crossbowVariableP) s->op->kernel->inputs[0];
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get labels */
	int labelsOffset = 0;
	crossbowVariableP theLabels = (crossbowVariableP) s->op->kernel->inputs[1];
	crossbowDataBufferP labels = crossbowVariableGetDataBuffer (s->labels, &labelsOffset, NULL);

	/* Get local variable */
	crossbowDataBufferP count = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, NULL, NULL);

	/* Get an output buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* This should be equal to the batch size */
	int elements = theLabels->schema->elements;
	int numberofclasses = theInput->schema->elements / elements;

#ifndef KERNEL_NOOP
	crossbowKernelAccuracyCompute<<<GET_BLOCKS(elements), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>
		(
			elements,
			numberofclasses,
			(float *) (input->dev), /* SoftMax probabilities */
			(int *) (labels->dev) + (labelsOffset / 4),
			(float *) (count->dev)
		);
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (elements);
	UNUSED (numberofclasses);
	UNUSED (input);
	UNUSED (labels);
	UNUSED (count);
#endif

#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSetPointerMode (s->cublasHandle[s->op->branch], CUBLAS_POINTER_MODE_DEVICE));
	checkCublasStatus(cublasSasum (s->cublasHandle[s->op->branch], elements, (float *) (count->dev), 1, (float *) (output->dev)));
	checkCublasStatus(cublasSetPointerMode(s->cublasHandle[s->op->branch], CUBLAS_POINTER_MODE_HOST));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (elements);
	UNUSED (count);
	UNUSED (output);
#endif

	/* Store output */
	crossbowListAppend (s->outputs[s->op->id], output);

	/* Return read-write local variable to kernel when the dataflow execution completes */
	crossbowListAppend (s->locals[s->op->id], count);

	return;
}
