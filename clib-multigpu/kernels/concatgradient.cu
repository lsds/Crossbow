#include "concatgradient.h"

__global__ void crossbowKernelConcatGradientCompute (
		const int count,
		const float* input,
		const int size,
		const int outputChannels,
		const int inputChannels,
		const int offset,
		float* output) {

	CUDA_KERNEL_LOOP(index, count) {

		const int total_concat_size = size * inputChannels;
		const int concat_num = index / total_concat_size;
		const int concat_index = index % total_concat_size;
		const int top_index = concat_index + (concat_num * outputChannels + offset) * size;

		output[index] = input[top_index];
	}
}

void crossbowKernelConcatGradient (void *args) {

	int i;

	crossbowStreamP s = (crossbowStreamP) args;

	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get kernel configuration parameters */
	int axis = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	int offset = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 1));

	int channelOffset = 0;
	for (i = 0; i < offset; i++)
		channelOffset += crossbowVariableSchemaShape (s->op->peer->kernel->inputs[i]->schema, axis);

	/* Get an output buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Fill output buffer with zeros */
#ifndef CUDART_NOOP
	checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (output);
#endif

	int concats = crossbowVariableSchemaCountElementsInRange (s->op->kernel->inputs[0]->schema, 0, axis);
	int size = crossbowVariableSchemaCountElementsFrom (s->op->kernel->inputs[0]->schema, axis + 1);

	int outputChannels = crossbowVariableSchemaShape (s->op->kernel->output->schema, axis);

	int inputChannels = crossbowVariableSchemaShape (s->op->kernel->inputs[0]->schema, axis);
	int inputElements = inputChannels * size;

	int threads = inputElements * concats;

#ifndef KERNEL_NOOP
	crossbowKernelConcatGradientCompute<<< GET_BLOCKS(threads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch] >>>(
				threads,
				(float *) (input->dev),
				size,
				outputChannels,
				inputChannels,
				channelOffset,
				(float *) (output->dev)
				);
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (threads);
	UNUSED (input);
	UNUSED (size);
	UNUSED (outputChannels);
	UNUSED (inputChannels);
	UNUSED (channelOffset);
	UNUSED (output);
#endif

	/* Append output to list */
	crossbowListAppend (s->outputs[s->op->id], output);

	return;
}
