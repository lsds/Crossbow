#include "cudnnconvgradient.h"

void crossbowCudnnKernelConvGradient (void *args) {

	float alpha, beta;

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get model variable `weights` and `bias` */
	int hasBias = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));

	/* Set model and local variables */
	crossbowDataBufferP weightGradient, biasGradient = NULL;
	int weightGradientOffset = 0, biasGradientOffset = 0;
	int weightGradientLength = 0, biasGradientLength = 0;

	/*
	 * The following implements the parameter server synchronisation model
	 *
	 * The GPU worker should wait for the application of the previously computed gradient (if any)
	 * to the parameter server model (scheduled on a different stream) to complete.
	 */
#ifdef UPDATE_MODEL_INCREMENTALLY
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server[s->op->peer->id], 0));
#else
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server, 0));
#endif

	weightGradient = crossbowModelVariableGradient (s->model, s->op->peer->kernel->id, 1, &weightGradientOffset, &weightGradientLength);
	if (hasBias)
		biasGradient = crossbowModelVariableGradient (s->model, s->op->peer->kernel->id, 2, &biasGradientOffset, &biasGradientLength);
    
    /* Fill gradient buffer with zeros */
#ifndef CUDART_NOOP
	cudaMemsetAsync ((void *)(((char *) weightGradient->dev) + weightGradientOffset), 0, weightGradientLength + biasGradientLength, s->stream[s->op->branch]);
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (weightGradient);
    UNUSED (weightGradientOffset);
    UNUSED (weightGradientLength);
    UNUSED (biasGradientLength);
#endif

	nullPointerException(s->op->peer);

	/* Get input of peer operator */
	crossbowDataBufferP peerInput = crossbowStreamGetPeerInput (s);

	/* Get cuDNN pooling parameters (from peer operator) */
	crossbowCudnnConvParamsP params = s->op->peer->kernel->descriptors.conv;

	/* Get workspace variable */
	int backwardFilterWorkSpaceSizeInBytes = 0;
	crossbowDataBufferP backwardFilterWorkSpace = NULL;

	if (params->backwardFilterWorkSpaceSize > 0)
		backwardFilterWorkSpace = crossbowLocalVariableGetDataBuffer (
            (crossbowLocalVariableP) crossbowArrayListGet (s->op->peer->kernel->variables, 1), 
                s->deviceId, s->id, NULL, &backwardFilterWorkSpaceSizeInBytes);

	/* Compute gradient with respect to bias */

	if (hasBias) {

		alpha = 1;
		beta = 1;

#ifndef CUDANN_NOOP
		checkCudnnStatus(cudnnConvolutionBackwardBias(s->cudnnHandle[s->op->branch],
				&alpha,
				params->output->descriptor,
				input->dev,
				&beta,
				params->biasDesc,
				(void *) ((char *) (biasGradient->dev) + biasGradientOffset)
				));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (alpha);
        UNUSED (params);
        UNUSED (input);
        UNUSED (beta);
        UNUSED (biasGradient);
        UNUSED (biasGradientOffset);
#endif

#ifdef COMPUTE_CHECKSUM
		float biasGradientChecksum = crossbowDataBufferComputeCheckSum (s->model->gradient, biasGradientOffset, biasGradientLength);
		info("Kernel's %s bias gradient checksum is %.5f\n", s->op->kernel->name, biasGradientChecksum);
#endif
	}

	/* Compute gradient with respect to weights */

	alpha = 1;
	beta = 1;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnConvolutionBackwardFilter(s->cudnnHandle[s->op->branch],
			&alpha,
			params->input->descriptor,
			peerInput->dev,
			params->output->descriptor,
			input->dev,
			params->convDesc,
			params->backwardFilterAlgorithm,
			(backwardFilterWorkSpace) ? backwardFilterWorkSpace->dev : NULL,
			backwardFilterWorkSpaceSizeInBytes,
			&beta,
			params->filterDesc,
			(void *) ((char *) (weightGradient->dev) + weightGradientOffset)
			));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (alpha);
    UNUSED (params);
    UNUSED (peerInput);
    UNUSED (input);
    UNUSED (backwardFilterWorkSpace);
    UNUSED (backwardFilterWorkSpaceSizeInBytes);
    UNUSED (beta);
    UNUSED (weightGradient);
    UNUSED (weightGradientOffset);
#endif

	/* Compute input data gradient */

	int backwardDataWorkSpaceSizeInBytes = 0;
	crossbowDataBufferP backwardDataWorkSpace = NULL;

	if (! crossbowDataflowMostUpstream (s->dataflow, s->op->peer)) {

		/* Get an output variable buffer */
		crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

		int weightOffset = 0;
#ifdef TRAIN_WITH_MASTER
		crossbowDataBufferP weight = crossbowModelVariable (s->theModel, s->op->peer->kernel->id, 1, &weightOffset, NULL);
#else
		crossbowDataBufferP weight = crossbowModelVariable (s->model,    s->op->peer->kernel->id, 1, &weightOffset, NULL);
#endif

		/* Get workspace variable */
		if (params->backwardDataWorkSpaceSize > 0)
			backwardDataWorkSpace = crossbowLocalVariableGetDataBuffer (
                (crossbowLocalVariableP) crossbowArrayListGet (s->op->peer->kernel->variables, 2), 
                    s->deviceId, s->id, NULL, &backwardDataWorkSpaceSizeInBytes);

		alpha = 1;
		beta = 0;

#ifndef CUDANN_NOOP
		checkCudnnStatus(cudnnConvolutionBackwardData(s->cudnnHandle[s->op->branch],
				&alpha,
				params->filterDesc,
				(void *) ((char *) (weight->dev) + weightOffset),
				params->output->descriptor,
				input->dev,
				params->convDesc,
				params->backwardDataAlgorithm,
				(backwardDataWorkSpace) ? backwardDataWorkSpace->dev : NULL,
			    backwardDataWorkSpaceSizeInBytes,
				&beta,
				params->input->descriptor,
				output->dev
				));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (alpha);
        UNUSED (params);
        UNUSED (weight);
        UNUSED (weightOffset);
        UNUSED (input);
        UNUSED (backwardDataWorkSpace);
        UNUSED (backwardDataWorkSpaceSizeInBytes);
        UNUSED (beta);
        UNUSED (output);
#endif

		/* Store output in stream */
		crossbowListAppend(s->outputs[s->op->id], output);

	}

	/* Return read-write local variables to kernel when the dataflow execution completes */

	if (backwardFilterWorkSpace)
		crossbowListAppend (s->locals[s->op->id], backwardFilterWorkSpace);

	if (backwardDataWorkSpace)
		crossbowListAppend (s->locals[s->op->id], backwardDataWorkSpace);

	return;
}
