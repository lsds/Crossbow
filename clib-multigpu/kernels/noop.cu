#include "noop.h"

void crossbowKernelNoop (void *args) {

	/* GEMM variables */
	float *A, *B, *C;
	int M, N, K;
	float alpha, beta;

	crossbowStreamP s = (crossbowStreamP) args;

	int offset, length;
	crossbowDataBufferP unit = crossbowModelVariable (s->model, s->op->kernel->id, 1, &offset, &length);

	crossbowDataBufferP input  = crossbowStreamGetCurrentInput (s);
	crossbowDataBufferP output = crossbowStreamGetCurrentOutput (s);

	/* Get kernel configuration parameter */
	int axis = crossbowKernelConfigParamGetIntValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));

	/* Set device buffers */
	A = (float *) (input->dev);
	B = (float *) (unit->dev);
	C = (float *) (output->dev);
	M = crossbowVariableSchemaCountElementsInRange (s->examples->schema, 0, axis);
	K = crossbowVariableSchemaCountElementsFrom    (s->examples->schema, axis);
	N = K;
	alpha = 1;
	beta = 0;

#ifndef CUBLAS_NOOP
	checkCublasStatus(cublasSgemm (s->cublasHandle[s->op->branch], CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M));
#else
	/* Subterfuge unused parameter warnings */
	UNUSED (M);
	UNUSED (N);
	UNUSED (K);
	UNUSED (alpha);
	UNUSED (A);
	UNUSED (B);
	UNUSED (beta);
	UNUSED (C);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);
}
