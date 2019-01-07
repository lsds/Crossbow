#include "innerproduct.h"

void crossbowKernelInnerProduct (void *args) {

	/* Kernel configuration parameters */
	int axis;
	int outputs; /* Number of outputs (= number of examples in batch) */
	int hasBias;

	/* BLAS parameters */
	int M, N, K;
	float alpha, beta;
	float *A, *B, *C;
	cublasOperation_t transA, transB;
	int lda, ldb, ldc;

	/* Input and output variables */
	crossbowDataBufferP input, output;

	/* Model variables */
	crossbowDataBufferP weights, bias;
	int weight_offset;
	int bias_offset;

	int weight_length, bias_length;

	/* Local variables */
	crossbowDataBufferP multiplier = NULL;

	crossbowStreamP s = (crossbowStreamP) args;
	/* checkCublasStatus(cublasSetStream (s->cublasHandle, s->stream)); */

	/* Get kernel configuration parameters */
	   axis = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	outputs = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 1));
	hasBias = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 2));

	/*
	 * The following implements the parameter server
	 * synchronisation model
	 */

	/* The GPU worker should wait for the model to be updated with the latest gradients */
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));

	/* Get model and related local variables (bias multiplier)
	 *
	 * `weights` and `bias` share the same data buffer since models
	 * are stored in contiguous regions of memory. `weights` is the
	 * first variable, so its offset is 0.
	 *
	 * The bias multiplier `multiplier` is a read-only variable.
	 */
	if (s->phi == CHECK) {
#ifdef CHECK_WITH_MASTER
		weights = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 1, &weight_offset, &weight_length, s->cublasHandle[s->op->branch]);
#else
		weights = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 1, &weight_offset, &weight_length, s->cublasHandle[s->op->branch]);
#endif
	}
	else {
#ifdef TRAIN_WITH_MASTER
		weights = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 1, &weight_offset, &weight_length, s->cublasHandle[s->op->branch]);
#else
		weights = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 1, &weight_offset, &weight_length, s->cublasHandle[s->op->branch]);
#endif
	}
		
	if (hasBias) {
		if (s->phi == CHECK) {
#ifdef CHECK_WITH_MASTER
			bias = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 2, &bias_offset, &bias_length, s->cublasHandle[s->op->branch]);
#else
			bias = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 2, &bias_offset, &bias_length, s->cublasHandle[s->op->branch]);
#endif
		}
		else {
#ifdef TRAIN_WITH_MASTER
			bias = crossbowModelVariableAccelerated (s->theModel, s->op->kernel->id, 2, &bias_offset, &bias_length, s->cublasHandle[s->op->branch]);
#else
			bias = crossbowModelVariableAccelerated (s->model,    s->op->kernel->id, 2, &bias_offset, &bias_length, s->cublasHandle[s->op->branch]);
#endif
		}

		multiplier = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, NULL, NULL);
	}
	
	/* Debugging... Writing weights to a file every 100 tasks.
	 *
	 * if (((s->task - 1) == 0) || ((s->task - 1) % 100 == 0)) {
	 * 		crossbowDataBufferWriteToFile (
	 * 			s->model->data, 
	 *			"/mnt/nfs/users/piwatcha/16-crossbow/scripts/msr/resnet-50/ip-weights", 
	 *			s->task,
	 * 			weight_offset, weight_length);
	 * }
	 */
    
#ifdef COMPUTE_CHECKSUM
	/* Debug model */
	float checksum;
	checksum = crossbowDataBufferComputeCheckSum (s->model->data, weight_offset, weight_length);
	info("Kernel's %s weights checksum is %.5f\n", s->op->kernel->name, checksum);
	if (hasBias) {
		checksum = crossbowDataBufferComputeCheckSum (s->model->data, bias_offset, bias_length);
		info("Kernel's %s bias checksum is %.5f\n", s->op->kernel->name, checksum);
	}
#endif

	/* Set input buffer */
	input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	output = crossbowStreamGetCurrentOutput (s);

	/* Set cuBLAS GEMM parameters */
	M = crossbowVariableSchemaCountElementsInRange (s->op->kernel->inputs[0]->schema, 0, axis);
	N = outputs;
	K = crossbowVariableSchemaCountElementsFrom (s->op->kernel->inputs[0]->schema, axis);

	alpha = 1;
	beta = 0;
	A = (float *) input->dev;
	B = (float *) (weights->dev) + (weight_offset / 4);
	C = (float *) output->dev;
	transA = CUBLAS_OP_N;
	transB = CUBLAS_OP_T;
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

	if (hasBias) {
		/* M and N are the same as before */
		K = 1;
		alpha = 1;
		beta = 1;
		A = (float *) multiplier->dev;
		/*
		 * TODO Compute offset based on byte length of elements in buffer
		 */
		B = (float *) (bias->dev) + (bias_offset / 4);
		C = (float *) output->dev;
		transA = CUBLAS_OP_N;
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
    }

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	/* There are not any read-write local variables to return */

	return;
}
