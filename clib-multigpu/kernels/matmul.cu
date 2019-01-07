#include "matmul.h"

void crossbowKernelMatMul (void *args) {

	/* GEMM variables */
	float *A, *B, *C;
	int M, N, K;
	float alpha, beta;

	crossbowDataBufferP __model_variable, output;
	int offset, length;

	crossbowStreamP s = (crossbowStreamP) args;

	__model_variable = crossbowModelVariable (s->model, s->op->kernel->id, 1, &offset, &length);
	output = crossbowStreamGetCurrentOutput (s);

	M = crossbowVariableSchemaCountElementsInRange (s->examples->schema, 0, 1);
	K = crossbowVariableSchemaCountElementsFrom    (s->examples->schema, 1);
	N = K;

	alpha = 1;
	beta = 0;

//	/* Set device buffers */
	A = (float *) s->input->dev; /* Examples are always the first variable */
	B = (float *) __model_variable->dev;
	C = (float *) output->dev;

	checkCublasStatus(cublasSgemm (s->cublasHandle[s->op->branch], CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M));

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);
	return;
}
