#include "softmaxlossgradient.h"

#include "../debug.h"

__global__ void crossbowKernelSoftMaxLossGradientCompute (
	const int nthreads,
	int* label,
	float* bottom_diff,
	const int num,
	const int dim,
	const int spatial_dim,
	const bool has_ignore_label_,
	const int ignore_label_,
	float* counts) {

	const int channels = dim / spatial_dim;

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = label[n * spatial_dim + s]; // static_cast<int>(label[n * spatial_dim + s]);

		if (has_ignore_label_ && label_value == ignore_label_) {
			for (int c = 0; c < channels; ++c) {
				bottom_diff[n * dim + c * spatial_dim + s] = 0;
			}
			counts[index] = 0;
		} else {
			bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
			counts[index] = 1;
		}
	}
}

void crossbowKernelSoftMaxLossGradient (void *args) {

	float alpha;

	crossbowVariableP peerInput, theInput, theLabels;
	crossbowDataBufferP peer_input, labels, output, counts;

	int labels_offset, counts_offset;
	int labels_length, counts_length, peer_input_length;

	int outer, inner, dim, ignorelabelvalue, nthreads;
	bool ignorelabel;

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input from previous operator */
	theInput  = (crossbowVariableP) s->op->kernel->inputs[0];
	theLabels = (crossbowVariableP) s->op->kernel->inputs[1];
#ifdef CROSSBOW_EXCEPTIONS
	nullPointerException(theInput);
#else
    UNUSED (theInput);
#endif
	nullPointerException(theLabels);
	nullPointerException(s->op->peer);

	peerInput = (crossbowVariableP) s->op->peer->kernel->inputs[0];

	peer_input = crossbowStreamGetPeerInput (s);

	peer_input_length = peerInput->schema->bytes;

	/* Get labels */
	labels = crossbowVariableGetDataBuffer (s->labels, &labels_offset, &labels_length);

	/* Get output */
	output = crossbowStreamGetCurrentOutput (s);

	/* Get local variable */
	counts = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, &counts_offset, &counts_length);

#ifndef CUDART_NOOP
	cudaMemcpyAsync (output->dev, peer_input->dev, peer_input_length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (output);
    UNUSED (peer_input);
    UNUSED (peer_input_length);
#endif
    
	nthreads = theLabels->schema->elements;
	outer = crossbowVariableSchemaCountElementsInRange (peerInput->schema, 0, 1);
	inner = crossbowVariableSchemaCountElementsFrom (peerInput->schema, 2);
	dim = peerInput->schema->elements / outer;

	ignorelabelvalue = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	ignorelabel = (ignorelabelvalue >= 0);
#ifndef KERNEL_NOOP
	crossbowKernelSoftMaxLossGradientCompute<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>(
        nthreads, 
        (int *) (labels->dev) + (labels_offset / 4), 
        (float *) (output->dev), 
        outer, 
        dim, 
        inner, 
        ignorelabel, 
        ignorelabelvalue, 
        (float *) (counts->dev));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (nthreads);
    UNUSED (labels);
    UNUSED (labels_offset);
    UNUSED (output);
    UNUSED (outer);
    UNUSED (dim);
    UNUSED (inner);
    UNUSED (ignorelabel);
    UNUSED (ignorelabelvalue);
    UNUSED (counts);
#endif

/* NOTE: Why is the following operation required? 
 * Is it some left-over from previous versions of
 * the code? 
 */
#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSetStream (s->cublasHandle[s->op->branch], s->stream[s->op->branch]));
#else
    /* Subterfuge unused parameter warnings */
#endif
    
	alpha = 1 / (float) outer;

#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSscal (s->cublasHandle[s->op->branch], s->op->kernel->output->schema->elements, &alpha, (float *) output->dev, 1));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (alpha);
    UNUSED (output);
#endif
    
	/* crossbowDataBufferPull (output, s->stream); */

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	crossbowListAppend (s->locals[s->op->id], counts);

	return;
}
