#include "concat.h"

__global__ void crossbowKernelConcatCompute (
		const int count,
		const float* input,
		const int size,
		const int outputChannels,
		const int inputChannels,
		const int offset,
		float* output) {

	CUDA_KERNEL_LOOP(index, count) {

		const int totalSize = size * inputChannels;
		const int concat_num = index / totalSize;
		const int concat_index = index % totalSize;
		const int top_index = concat_index + (concat_num * outputChannels + offset) * size;

		output[top_index] = input[index];
	}
}

void crossbowKernelConcat (void *args) {

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get output buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Fill output buffer with zeros */
#ifndef CUDART_NOOP
	checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (output);
#endif

	/* Get kernel configuration parameter */
	int axis = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));

	/* Find number of upstream operators */
	int numberOfInputs = crossbowArrayListSize (s->op->upstream);

	int concats = crossbowVariableSchemaCountElementsInRange (s->op->kernel->inputs[0]->schema, 0, axis);
	int size = crossbowVariableSchemaCountElementsFrom (s->op->kernel->inputs[0]->schema, axis + 1);

	int outputChannels = crossbowVariableSchemaShape (s->op->kernel->output->schema, axis);

	int offset = 0;

	for (int i = 0; i < numberOfInputs; ++i) {

		crossbowOperatorP prev = (crossbowOperatorP) crossbowArrayListGet (s->op->upstream, i);
		dbg("Concat operator input #%d is from operator %d (kernel %s)\n", i, prev->id, prev->kernel->name);
		/* Get output of previous operator */
		crossbowDataBufferP input = crossbowStreamOperatorGetOutput (s, prev);

		int inputChannels = crossbowVariableSchemaShape (s->op->kernel->inputs[i]->schema, axis);
		int inputElements = inputChannels * size;

		int threads = inputElements * concats;

#ifndef KERNEL_NOOP
		crossbowKernelConcatCompute<<< GET_BLOCKS(threads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch] >>>(
				threads,
				(float *) (input->dev),
				size,
				outputChannels,
				inputChannels,
				offset,
				(float *) (output->dev)
				);
#else
		/* Subterfuge unused parameter warnings */
		UNUSED (threads);
		UNUSED (input);
		UNUSED (size);
		UNUSED (outputChannels);
		UNUSED (inputChannels);
		UNUSED (offset);
		UNUSED (output);
#endif

		offset += inputChannels;
	}

	crossbowListAppend (s->outputs[s->op->id], output);

	return;
}
