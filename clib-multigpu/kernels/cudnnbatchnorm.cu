#include "cudnnbatchnorm.h"

void crossbowCudnnKernelBatchNorm (void *args) {

	float alpha, beta;
	double factor;

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	crossbowCudnnBatchNormParamsP params = s->op->kernel->descriptors.batchnorm;

	/* Get local variables */
	crossbowDataBufferP saveMean     = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, NULL, NULL);
	crossbowDataBufferP saveVariance = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 1), s->deviceId, s->id, NULL, NULL);

	/* Get buffers from BathNorm configuration */

	int updates = 0;
	crossbowDataBufferP runningMean     = NULL;
	crossbowDataBufferP runningVariance = NULL;

	crossbowCudnnBatchNormParamsGetEstimatedMeanAndVariable (params, s->deviceId, s->stream[s->op->branch], &runningMean, &runningVariance, &updates);
#ifdef COMPUTE_CHECKSUM
	float csum1, csum2, csum3, csum4, csum5, csum6;
#endif
	/* Get kernel configuration parameters */
	
	double epsilon = (double) crossbowKernelConfigParamGetDoubleValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 1));
	epsilon = max (epsilon, CUDNN_BN_MIN_EPSILON);
	/* epsilon =  1e-5; */
	/* epsilon = 0.001; */

	int cma = (int) crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 3));
	if (cma > 0) {
		/* Configure averaging factor for CMA */
		factor = 1. / (1. + updates);
	}
	else {
		double fraction = (double) crossbowKernelConfigParamGetDoubleValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 2));
		factor = 1 - fraction;
	}
	/* factor = 0.10; */
	/* factor = 0.01; */
	
	dbg("epsilon = %.5f, factor = %.5f\n", epsilon, factor);

	/* Get model variable `scale` and `bias` */
	crossbowDataBufferP scale, bias = NULL;
	int scaleOffset, biasOffset;
	int scaleLength, biasLength;

	/* The GPU worker should wait for the model to be updated with the latest gradients */
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));
	
	if (s->phi == CHECK) {
#ifdef CHECK_WITH_MASTER
		scale = crossbowModelVariable (s->theModel, s->op->kernel->id, 1, &scaleOffset, &scaleLength);
		bias  = crossbowModelVariable (s->theModel, s->op->kernel->id, 2,  &biasOffset,  &biasLength);
#else
		scale = crossbowModelVariable (s->model,    s->op->kernel->id, 1, &scaleOffset, &scaleLength);
		bias  = crossbowModelVariable (s->model,    s->op->kernel->id, 2,  &biasOffset,  &biasLength);
#endif
	} 
	else {
#ifdef TRAIN_WITH_MASTER
		scale = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 1, &scaleOffset, &scaleLength, s->cublasHandle[s->op->branch]);
		bias  = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 2,  &biasOffset,  &biasLength, s->cublasHandle[s->op->branch]);
#else
		scale = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 1, &scaleOffset, &scaleLength, s->cublasHandle[s->op->branch]);
		bias  = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 2,  &biasOffset,  &biasLength, s->cublasHandle[s->op->branch]);
#endif
	}
#ifdef COMPUTE_CHECKSUM
	/* Debug model */
	float checksum;
	checksum = crossbowDataBufferComputeCheckSum (s->model->data, scaleOffset, scaleLength);
	info("Kernel's %s scale checksum is %.5f\n", s->op->kernel->name, checksum);
	checksum = crossbowDataBufferComputeCheckSum (s->model->data, biasOffset, biasLength);
	info("Kernel's %s bias checksum is %.5f\n", s->op->kernel->name, checksum);
#endif
	
#ifdef GPU_VERBOSE
	char *str = crossbowCudnnBatchNormParamsString (params);
	dbg("%s %s\n", s->op->kernel->name, str);
	crossbowStringFree (str);
#endif

	alpha = 1;
	beta  = 0;

	if (s->phi == CHECK) {
#ifndef CUDANN_NOOP
		checkCudnnStatus(cudnnBatchNormalizationForwardInference(
			s->cudnnHandle[s->op->branch],
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			params->input->descriptor, input->dev,
			params->output->descriptor, output->dev,
			params->derivedBatchNormDesc,
			(void *) (((char *) (scale->dev)) + scaleOffset),
			(void *) (((char *) (bias->dev)) + biasOffset),
			runningMean->dev, runningVariance->dev,
			epsilon
			));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (alpha);
        UNUSED (beta);
        UNUSED (params);
        UNUSED (input);
        UNUSED (output);
        UNUSED (scale);
        UNUSED (scaleOffset);
        UNUSED (bias);
        UNUSED (biasOffset);
        UNUSED (runningMean);
        UNUSED (runningVariance);
        UNUSED (epsilon);
#endif

	} else {

#ifdef COMPUTE_CHECKSUM
		csum1 = crossbowDataBufferComputeCheckSum (runningMean,     0,     runningMean->size);
		csum2 = crossbowDataBufferComputeCheckSum (runningVariance, 0, runningVariance->size);
		info("Running mean and variance checksum before training are %.5f and %.5f\n", csum1, csum2);
#endif

#ifndef CUDANN_NOOP
        checkCudnnStatus(cudnnBatchNormalizationForwardTraining(
			s->cudnnHandle[s->op->branch],
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			params->input->descriptor, input->dev,
			params->input->descriptor, output->dev,
			params->derivedBatchNormDesc,
			(void *) (((char *) (scale->dev)) + scaleOffset),
    		(void *) (((char *) (bias->dev)) + biasOffset),
			factor,
			runningMean->dev, runningVariance->dev,
			epsilon,
			saveMean->dev, saveVariance->dev
			));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (alpha);
        UNUSED (beta);
        UNUSED (params);
        UNUSED (input);
        UNUSED (output);
        UNUSED (scale);
        UNUSED (scaleOffset);
        UNUSED (bias);
        UNUSED (biasOffset);
        UNUSED (factor);
        UNUSED (runningMean);
        UNUSED (runningVariance);
        UNUSED (epsilon);
        UNUSED (saveMean);
        UNUSED (saveVariance);
#endif

#ifdef COMPUTE_CHECKSUM
		csum3 = crossbowDataBufferComputeCheckSum (saveMean,     0,     saveMean->size);
		csum4 = crossbowDataBufferComputeCheckSum (saveVariance, 0, saveVariance->size);
		info("Save mean and variance checksum are %.5f and %.5f\n", csum3, csum4);

		csum5 = crossbowDataBufferComputeCheckSum (runningMean,     0,     runningMean->size);
		csum6 = crossbowDataBufferComputeCheckSum (runningVariance, 0, runningVariance->size);
		info("Running mean and variance checksum after training are %.5f (expected %.5f) and %.5f (expected %.5f).\n", 
            csum5, (csum3 * factor + csum1 * (1 - factor)), csum6, (csum4 * factor + csum2 * (1 - factor)));
#endif
	}

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	/* Return variables to kernel when the dataflow execution completes */
	crossbowCudnnBatchNormParamsReleaseEstimatedMeanAndVariable (params, s->deviceId, s->stream[s->op->branch]);

	crossbowListAppend (s->locals[s->op->id], saveMean);
	crossbowListAppend (s->locals[s->op->id], saveVariance);

	return;
}
