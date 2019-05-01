#include "cudnnconv.h"

#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>

void crossbowCudnnKernelConv (void *args) {

	float alpha, beta;

	crossbowStreamP s = (crossbowStreamP) args;

	/* Get input buffer */
	crossbowDataBufferP input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get model variable `weights` and `bias` */
	int hasBias = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));

	crossbowDataBufferP weight, bias = NULL;
	int weightOffset, biasOffset;
	int weightLength, biasLength;

	/* The GPU worker should wait for the model to be updated with the latest gradients */
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));
	
	if (s->phi == CHECK) {
#ifdef CHECK_WITH_MASTER
		weight = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 1, &weightOffset, &weightLength, s->cublasHandle[s->op->branch]);
#else
		weight = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 1, &weightOffset, &weightLength, s->cublasHandle[s->op->branch]);
#endif
	} 
	else {
#ifdef TRAIN_WITH_MASTER
		weight = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 1, &weightOffset, &weightLength, s->cublasHandle[s->op->branch]);
#else
		weight = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 1, &weightOffset, &weightLength, s->cublasHandle[s->op->branch]);
#endif
	}

	if (hasBias) {
		if (s->phi == CHECK) {
#ifdef CHECK_WITH_MASTER
			bias = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 2, &biasOffset, &biasLength, s->cublasHandle[s->op->branch]);
#else
			bias = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 2, &biasOffset, &biasLength, s->cublasHandle[s->op->branch]);
#endif
		} 
		else {
#ifdef TRAIN_WITH_MASTER
			bias = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 2, &biasOffset, &biasLength, s->cublasHandle[s->op->branch]);
#else
			bias = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 2, &biasOffset, &biasLength, s->cublasHandle[s->op->branch]);
#endif
		}
	}

#ifdef COMPUTE_CHECKSUM
	/* Debug model */
	float checksum;
	checksum = crossbowDataBufferComputeCheckSum (s->model->data, weightOffset, weightLength);
	info("Kernel's %s weights checksum is %.5f\n", s->op->kernel->name, checksum);
	if (hasBias) {
		checksum = crossbowDataBufferComputeCheckSum (s->model->data, biasOffset, biasLength);
		info("Kernel's %s bias checksum is %.5f\n", s->op->kernel->name, checksum);
	}
#endif

	crossbowCudnnConvParamsP params = s->op->kernel->descriptors.conv;
#ifdef GPU_VERBOSE
	char *str = crossbowCudnnConvParamsString (params);
	dbg("%s %s\n", s->op->kernel->name, str);
	crossbowStringFree (str);
#endif

	/* Get workspace */
	int workSpaceSizeInBytes = 0;
	crossbowDataBufferP workSpace = NULL;

	if (params->forwardWorkSpaceSize > 0) {

		workSpace = crossbowLocalVariableGetDataBuffer (
            (crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), 
                s->deviceId, s->id, NULL, &workSpaceSizeInBytes);
		nullPointerException(workSpace);
		dbg("Got workspace %p of size %d bytes\n", workSpace, workSpaceSizeInBytes);
	}

	alpha = 1;
	beta = 0;

#ifndef CUDANN_NOOP
	checkCudnnStatus(cudnnConvolutionForward(s->cudnnHandle[s->op->branch],
			&alpha,
			params->input->descriptor,
			input->dev,
			params->filterDesc,
			(void *) (((char *) (weight->dev)) + weightOffset),
			params->convDesc,
			params->forwardAlgorithm,
			(workSpace != NULL) ? workSpace->dev : NULL,
			workSpaceSizeInBytes, /* Must equal params->forwardWorkSpaceSize */
			&beta,
			params->output->descriptor,
			output->dev
			));
	
	if (hasBias) {

		alpha = 1;
		beta = 1;

		checkCudnnStatus(cudnnAddTensor(s->cudnnHandle[s->op->branch],
				&alpha,
				params->biasDesc,
				(void *) (((char *) (bias->dev)) + biasOffset),
				&beta,
				params->output->descriptor,
				output->dev
				));
	}
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (alpha);
    UNUSED (params);
    UNUSED (input);
    UNUSED (weight);
    UNUSED (workSpace);
    UNUSED (workSpaceSizeInBytes);
    UNUSED (beta);
    UNUSED (output);
    UNUSED (bias);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	/* Return read-write local variables to kernel when the dataflow execution completes */
	if (workSpace)
		crossbowListAppend (s->locals[s->op->id], workSpace);

	return;
}
