#include "cudnnbatchnormgradient.h"

void crossbowCudnnKernelBatchNormGradient (void *args) {

	float alpha, beta;

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get kernel configuration parameters */
	double epsilon = max(
        crossbowKernelConfigParamGetDoubleValue
            ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0)), 
        CUDNN_BN_MIN_EPSILON
        );
    dbg("epsilon = %.5f\n", epsilon);

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

	/* Get model variable gradients */
	crossbowDataBufferP scaleGradient, biasGradient = NULL;
	int scaleGradientOffset, biasGradientOffset;
	int scaleGradientLength, biasGradientLength;

	scaleGradient = crossbowModelVariableGradient (s->model, s->op->peer->kernel->id, 1, &scaleGradientOffset, &scaleGradientLength);
	biasGradient  = crossbowModelVariableGradient (s->model, s->op->peer->kernel->id, 2,  &biasGradientOffset, 	&biasGradientLength);

	/* Reset gradient(s) */
#ifndef CUDART_NOOP
	cudaMemsetAsync ((void *)(((char *) scaleGradient->dev) + scaleGradientOffset), 0, scaleGradientLength + biasGradientLength, s->stream[s->op->branch]);
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (scaleGradient);
    UNUSED (scaleGradientOffset);
    UNUSED (scaleGradientLength);
    UNUSED (biasGradientLength);
#endif
    
	/* Get input of peer operator */
	nullPointerException(s->op->peer);
	crossbowDataBufferP peerInput = crossbowStreamGetPeerInput (s);
	
	/* Get cuDNN parameters (from peer operator) */
	crossbowCudnnBatchNormParamsP params = s->op->peer->kernel->descriptors.batchnorm;
	
	/* Get scale */
	int scaleOffset = 0;
	int scaleLength;
#ifdef TRAIN_WITH_MASTER
	crossbowDataBufferP scale = crossbowModelVariable (s->theModel, s->op->peer->kernel->id, 1, &scaleOffset, &scaleLength);
#else
	crossbowDataBufferP scale = crossbowModelVariable (s->model,    s->op->peer->kernel->id, 1, &scaleOffset, &scaleLength);
#endif
	
	crossbowDataBufferP saveMean      = (crossbowDataBufferP) crossbowListPeek (s->locals[s->op->peer->id], 0); // 2);
	crossbowDataBufferP saveVariance  = (crossbowDataBufferP) crossbowListPeek (s->locals[s->op->peer->id], 1); // 3);

	alpha = 1;
	beta  = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnBatchNormalizationBackward(
			s->cudnnHandle[s->op->branch],
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			&alpha,
			&alpha,
			params->input->descriptor, peerInput->dev,
			params->output->descriptor, input->dev,
			params->input->descriptor, 	output->dev,
			params->derivedBatchNormDesc,
			(void *) (((char *) (scale->dev)) + scaleOffset),
			(void *) (((char *) (scaleGradient->dev)) + scaleGradientOffset),
			(void *) (((char *) (biasGradient->dev)) + biasGradientOffset),
			epsilon,
			saveMean->dev, saveVariance->dev
			));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (alpha);
    UNUSED (beta);
    UNUSED (params);
    UNUSED (peerInput);
    UNUSED (input);
    UNUSED (output);
    UNUSED (scale);
    UNUSED (scaleOffset);
    UNUSED (scaleGradient);
    UNUSED (scaleGradientOffset);
    UNUSED (biasGradient);
    UNUSED (biasGradientOffset);
    UNUSED (epsilon);
    UNUSED (saveMean);
    UNUSED (saveVariance);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
