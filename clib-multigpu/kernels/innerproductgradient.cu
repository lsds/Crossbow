#include "innerproductgradient.h"

void crossbowKernelInnerProductGradient (void *args) {

	(void) args;

	/* BLAS parameters */
	float *A, *B, *C;
	int M0, N0, K0;
	int M, N, K;
	float alpha, beta;
	int lda, ldb, ldc;
	cublasOperation_t transA, transB;

	/* Kernel configuration parameters */
	int axis;
	int outputs;
	int hasBias;

	crossbowDataBufferP input, peer_input, weightgradient, biasgradient, biasmultiplier = NULL;

	/* Variable offsets */
	// int peer_input_offset,
	int weights_offset, bias_offset, biasmultiplier_offset;
	// int peer_input_length,
	int weights_length = 0, bias_length = 0, biasmultiplier_length;

	crossbowStreamP s = (crossbowStreamP) args;
	/* checkCublasStatus(cublasSetStream (s->cublasHandle, s->stream)); */

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

	/* Set kernel configuration parameters */
	   axis = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	outputs = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 1));
	hasBias = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 2));

	/* Get gradient computed by the previous operator */
	input = (crossbowDataBufferP) crossbowStreamGetCurrentInput (s);
	nullPointerException(input);

	/* Get an output variable buffer */
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get input of peer operator */
	nullPointerException(s->op->peer);
	peer_input = crossbowStreamGetPeerInput (s);

	/* Set model and local variables */
	weightgradient = crossbowModelVariableGradient (s->model, s->op->peer->kernel->id, 1, &weights_offset, &weights_length);
	if (hasBias) {
		biasgradient = crossbowModelVariableGradient (s->model, s->op->peer->kernel->id, 2, &bias_offset, &bias_length);
		biasmultiplier = crossbowLocalVariableGetDataBuffer (
            (crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), 
                s->deviceId, s->id, &biasmultiplier_offset, &biasmultiplier_length);
	}

	M0 = crossbowVariableSchemaCountElementsInRange (s->op->kernel->output->schema, 0, axis);
	N0 = outputs;
	K0 = crossbowVariableSchemaCountElementsFrom (s->op->kernel->output->schema, axis);

	/* Compute weight gradient */
	M = N0;
	N = K0;
	K = M0;
	alpha = 1;
	beta = 0;
	A = (float *) input->dev;
	B = (float *) peer_input->dev;
	C = (float *) (weightgradient->dev) + (weights_offset / 4);
	transA = CUBLAS_OP_T;
	transB = CUBLAS_OP_N;
	lda = (transA == CUBLAS_OP_N) ? K : M;
	ldb = (transB == CUBLAS_OP_N) ? N : K;
	ldc = N;

#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSgemm (s->cublasHandle[s->op->branch], transB, transA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (transB);
    UNUSED (transA);
    UNUSED (N);
    UNUSED (M);
    UNUSED (K);
    UNUSED (alpha);
    UNUSED (B);
    UNUSED (ldb);
    UNUSED (A);
    UNUSED (lda);
    UNUSED (beta);
    UNUSED (C);
    UNUSED (ldc);
#endif
	/* Compute bias gradient */
	M = M0;
	N = N0;
	if (hasBias) {
		/* M and N are the same as before */
		K = 1;
		alpha = 1;
		// Note: why does beta has to be 1?
		beta = 0;
		C = (float *) (biasgradient->dev) + (bias_offset / 4);
#ifndef CUBLAS_NOOP
		checkCublasStatus(cublasSgemv (s->cublasHandle[s->op->branch], CUBLAS_OP_N, N, M, &alpha, A, N, (float *)(biasmultiplier->dev), 1, &beta, C, 1));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (N);
        UNUSED (M);
        UNUSED (alpha);
        UNUSED (A);
        UNUSED (biasmultiplier);
        UNUSED (beta);
        UNUSED (C);
#endif
    }

	if (! crossbowDataflowMostUpstream(s->dataflow, s->op->peer)) {

		int fwdweightoffset = 0;
#ifdef TRAIN_WITH_MASTER
		crossbowDataBufferP weights = crossbowModelVariable (s->theModel, s->op->peer->kernel->id, 1, &fwdweightoffset, NULL);
#else
		crossbowDataBufferP weights = crossbowModelVariable (s->model,    s->op->peer->kernel->id, 1, &fwdweightoffset, NULL);
#endif

		transA = CUBLAS_OP_N;
		transB = CUBLAS_OP_N;
		M = M0;
		N = K0;
		K = N0;
		lda = (transA == CUBLAS_OP_N) ? K : M;
		ldb = (transB == CUBLAS_OP_N) ? N : K;
		ldc = N;
		alpha = 1;
		beta = 0;
		A = (float *) (input->dev);
		B = (float *) (weights->dev) + (fwdweightoffset / 4);
		C = (float *) (output->dev);
#ifndef CUBLAS_NOOP
		checkCublasStatus(cublasSgemm (s->cublasHandle[s->op->branch], transB, transA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (transB);
        UNUSED (transA);
        UNUSED (N);
        UNUSED (M);
        UNUSED (K);
        UNUSED (alpha);
        UNUSED (B);
        UNUSED (ldb);
        UNUSED (A);
        UNUSED (lda);
        UNUSED (beta);
        UNUSED (C);
        UNUSED (ldc);
#endif
    }

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	return;
}
