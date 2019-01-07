#include "softmaxloss.h"

__global__ void crossbowKernelSoftMaxLossCompute (
	const int nthreads,
	const float* prob_data,
	const int* label,
	float* loss,
	const int num,
	const int dim,
	const int spatial_dim,
	const bool has_ignore_label_,
	const int ignore_label_,
	float* counts) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = label[n * spatial_dim + s];
		if (has_ignore_label_ && label_value == ignore_label_) {
			loss[index] = 0;
			counts[index] = 0;
		} else {
			loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s], FLT_MIN));
			counts[index] = 1;
		}
	}
}

void crossbowKernelSoftMaxLoss (void *args) {

	/* Kernel configuration parameters */
	int ignorelabelvalue;
	bool ignorelabel;

	int outer, inner, dim, nthreads;

	float alpha;
	float *C;

	/* Input and output variables */
	crossbowVariableP theInput, theLabels;
	crossbowDataBufferP input, labels;
	crossbowDataBufferP output;
	int labels_offset;

	/* Local variables */
	crossbowDataBufferP losses, counts;
	int losses_length, counts_length;
	
	/* struct cudaPointerAttributes attributes; */
	
	crossbowStreamP s = (crossbowStreamP) args;

	/* checkCublasStatus(cublasSetStream (s->cublasHandle, s->stream)); */

	/* Get input variable */
	theInput = (crossbowVariableP) s->op->kernel->inputs[0];

	if (crossbowDataflowMostUpstream(s->dataflow, s->op))
		illegalStateException();

	input = crossbowStreamGetCurrentInput (s);

	/* Get labels */
	theLabels = (crossbowVariableP) s->op->kernel->inputs[1];
	labels = crossbowVariableGetDataBuffer (s->labels, &labels_offset, NULL);
	
	/*
	checkCudaErrors(cudaPointerGetAttributes (&attributes, labels->dev));
	info("labels->dev at %p: device %d device pointer %p host pointer %p managed %d\n", 
        labels->dev, attributes.device, attributes.devicePointer, attributes.hostPointer, attributes.isManaged);
	*/
	
	/* Get read-write local variables */
	losses = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, NULL, &losses_length);
	counts = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 1), s->deviceId, s->id, NULL, &counts_length);

	/* Get an output buffer */
	output = crossbowStreamGetCurrentOutput (s);

	/* Get kernel configuration parameters */
	ignorelabelvalue = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	ignorelabel = (ignorelabelvalue >= 0);

	nthreads = theLabels->schema->elements;
	outer = crossbowVariableSchemaCountElementsInRange (theInput->schema, 0, 1);
	inner = crossbowVariableSchemaCountElementsFrom (theInput->schema, 2);
	dim = theInput->schema->elements / outer;

#ifndef KERNEL_NOOP
	crossbowKernelSoftMaxLossCompute<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>(
        nthreads, 
        (float *) (input->dev), 
        (int *) (labels->dev) + (labels_offset / 4), 
        (float *) (losses->dev), 
        outer, 
        dim, 
        inner, 
        ignorelabel, 
        ignorelabelvalue, 
        (float *) (counts->dev));
#else
    /* Subterfuge unused parameter warnings */
	UNUSED (nthreads);
    UNUSED (input);
    UNUSED (labels);
    UNUSED (labels_offset);
    UNUSED (losses);
    UNUSED (outer);
    UNUSED (dim);
    UNUSED (inner);
    UNUSED (ignorelabel);
    UNUSED (ignorelabelvalue);
    UNUSED (counts);
#endif

	alpha = 1 / (float) nthreads;
#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSscal (s->cublasHandle[s->op->branch], nthreads, &alpha, (float *) losses->dev, 1));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (nthreads);
    UNUSED (alpha);
    UNUSED (losses);
#endif
    
	C = (float *) output->dev;
	
	/*
	checkCudaErrors(cudaPointerGetAttributes (&attributes, output->dev));
	
    dbg("output->dev at %p: device %d device pointer %p host pointer %p managed %d\n", 
        output->dev, attributes.device, attributes.devicePointer, attributes.hostPointer, attributes.isManaged);
	
    checkCudaErrors(cudaPointerGetAttributes (&attributes, losses->dev));
	
    dbg("losses->dev at %p: device %d device pointer %p host pointer %p managed %d\n", 
        losses->dev, attributes.device, attributes.devicePointer, attributes.hostPointer, attributes.isManaged);
	*/

#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSetPointerMode(s->cublasHandle[s->op->branch], CUBLAS_POINTER_MODE_DEVICE));
	checkCublasStatus(cublasSasum (s->cublasHandle[s->op->branch], nthreads, (float *) losses->dev, 1, C));
	checkCublasStatus(cublasSetPointerMode(s->cublasHandle[s->op->branch], CUBLAS_POINTER_MODE_HOST));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (nthreads);
    UNUSED (losses);
    UNUSED (C);
#endif
    
	/* Store output in stream */
	crossbowListAppend (s->outputs[s->op->id], output);

	/* Return read-write local variables to kernel when the dataflow execution completes */
	crossbowListAppend (s->locals[s->op->id], counts);
	crossbowListAppend (s->locals[s->op->id], losses);

	return;
}
