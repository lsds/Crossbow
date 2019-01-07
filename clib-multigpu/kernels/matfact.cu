#include "matfact.h"

/*
 * Each thread is assigned an (i,j) cell
 */
__global__ void crossbowKernelMatFactCompute (
	const int cells,
	const int K,
	const float lambda,
	const float rate,
	const int *input,
	const float *labels,
	const float *A0, /* model variables */
	const float *B0,
	float *A1, /* model gradients */
	float *B1,
	float *losses
	) {

	/* Get thread id */
	CUDA_KERNEL_LOOP(ndx, cells) {

	if (ndx >= cells)
		break;

	int row = input[ndx * 2 + 0];
	int col = input[ndx * 2 + 1];

	float rating = labels[ndx];

	/* Get row `row` from A0 and column `col` from B0
	 *
	 * A0 is N x K
	 * B0 is M x K
	 */
	float product = 0;
	for (int k = 0; k < K; k++) {
		float u = A0[row * K + k];
		float v = B0[col * K + k];
		product += (u * v);
	}

	float error = rating - product;
	float loss = error * error;

	losses[ndx] = loss;

	/* Update gradient */
	for (int k = 0; k < K; k++) {
		/* Get model values */
		float u = A0[row * K + k];
		float v = B0[col * K + k];
		/* Atomically update gradient */
		atomicAdd (&(A1[row * K + k]), (-(rate * ((2 * error * v) - (2 * lambda * u)))));
		atomicAdd (&(B1[col * K + k]), (-(rate * ((2 * error * u) - (2 * lambda * v)))));
	}

	} /* End of CUDA_KERNEL_LOOP */

	return;
}

void crossbowKernelMatFact (void *args) {

	/* Model variables (*0) and their gradients (*1) */
	crossbowDataBufferP A0, B0, A1, B1;

	/* Model variable and gradient offsets and lengths */
	int a0_offset, a0_length;
	int b0_offset, b0_length;
	int a1_offset, a1_length;
	int b1_offset, b1_length;

	/* Input and output variables */
	crossbowVariableP theInput, theLabels;
	crossbowDataBufferP input, labels, output;

	/* Input and output variable offsets and lengths */
	int input_offset, labels_offset;
	int input_length, labels_length;

	/* Kernel configuration arguments */
	// int M, N;
	int K;
	float lambda, rate;

	float one = -1;

	/* Local kernel variables */
	crossbowDataBufferP losses;

	/* Local kernel variable offsets and lengths */
	int losses_offset, losses_length;

	crossbowStreamP s = (crossbowStreamP) args;

	/* Init */

	/* The GPU worker should wait for the model to be updated with the latest gradients */
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));

	/* The GPU worker should wait for the application of the previously computed gradient (if any)
	 * to the parameter server model (scheduled on a different stream) to complete.
	 */
#ifdef UPDATE_MODEL_INCREMENTALLY
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server[s->op->id], 0));
#else
	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server, 0));
#endif
	/* Get model variables */
	A0 = crossbowModelVariable (s->model, s->op->kernel->id, 1, &a0_offset, &a0_length);
	B0 = crossbowModelVariable (s->model, s->op->kernel->id, 2, &b0_offset, &b0_length);

	dbg("Model variable #1 %p (offset %d) %d bytes\n", A0, a0_offset, a0_length);
	dbg("Model variable #2 %p (offset %d) %d bytes\n", B0, b0_offset, b0_length);

	/* Get model gradients */
	A1 = crossbowModelVariableGradient (s->model, s->op->kernel->id, 1, &a1_offset, &a1_length);
	B1 = crossbowModelVariableGradient (s->model, s->op->kernel->id, 2, &b1_offset, &b1_length);

	/* Reset the entire gradient buffer (A1 and B1 are consecutive in memory) */
	cudaMemsetAsync (s->model->gradient->dev, 0, s->model->bytes, s->stream[s->op->branch]);

	dbg("Model gradient variable #1 %p (offset %d) %d bytes\n", A1, a1_offset, a1_length);
	dbg("Model gradient variable #2 %p (offset %d) %d bytes\n", B1, b1_offset, b1_length);

	/* Get input */
	theInput = (crossbowVariableP) s->op->kernel->inputs[0];

	if (! crossbowDataflowMostUpstream(s->dataflow, s->op))
		illegalStateException();
	input = crossbowVariableGetDataBuffer (s->examples, &input_offset, &input_length); /* offset should be 0 */

	/* Get labels */
	theLabels = (crossbowVariableP) s->op->kernel->inputs[1];

	labels = crossbowVariableGetDataBuffer (s->labels, &labels_offset, &labels_length);

	char *theInputStr  = crossbowVariableString(theInput);
	char *theLabelsStr = crossbowVariableString(theLabels);

	dbg("Examples %p (offset %d) %d bytes %s\n",  input,  input_offset,  input_length,  theInputStr);
	dbg("Labels   %p (offset %d) %d bytes %s\n", labels, labels_offset, labels_length, theLabelsStr);

	crossbowStringFree (theInputStr);
	crossbowStringFree (theLabelsStr);

	/* Get an output variable */
	output = crossbowStreamGetCurrentOutput (s);
	/* Reset output buffer */
	checkCudaErrors(cudaMemsetAsync(output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));

	/* Get local variable(s) */
	losses = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, 0), s->deviceId, s->id, &losses_offset, &losses_length);
	/* Reset read-write local variables */
	checkCudaErrors(cudaMemsetAsync(losses->dev, 0, losses_length, s->stream[s->op->branch]));

	dbg("Local variable `losses` %p (offset %d) %d bytes\n", losses, losses_offset, losses_length);

	/* Get kernel configuration argument(s) */
	K      = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	/*
	 * M      = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 1));
	 * N      = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 2));
	 */
	lambda = crossbowKernelConfigParamGetFloatValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 3));
	rate   = crossbowKernelConfigParamGetFloatValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 4));

	/*
	 * dbg("%d rows, %d columns, %d latent variables\n", M, N, K);
	 */
	dbg("lambda = %.4f rate = %.4f\n", lambda, rate);

	/* Cells should equal the batch size */
	int cells = crossbowVariableSchemaCountElementsInRange (theInput->schema, 0, 1);

	dbg("%d cells\n", cells);
	dbg("B0 offset %d B1 offset %d\n", (b0_offset / 4), (b1_offset / 4));

	/* Compute */

	crossbowKernelMatFactCompute<<<GET_BLOCKS(cells), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>(
			cells,
			K,
			lambda,
			rate,
			(int *) (input->dev),
			(float *) (labels->dev) + (labels_offset / 4),
			(float *) (A0->dev),
			(float *) (B0->dev) + (b0_offset / 4),
			(float *) (A1->dev),
			(float *) (B1->dev) + (b1_offset / 4),
			(float *) (losses->dev)
	);

	/* Sum the loss */
	float *C = (float *) (output->dev);

	checkCublasStatus(cublasSetPointerMode(s->cublasHandle[s->op->branch],CUBLAS_POINTER_MODE_DEVICE));
	checkCublasStatus(cublasSasum (s->cublasHandle[s->op->branch], cells, (float *) (losses->dev), 1, C));
	checkCublasStatus(cublasSetPointerMode(s->cublasHandle[s->op->branch],CUBLAS_POINTER_MODE_HOST));

#ifdef UPDATE_MODEL_INCREMENTALLY
	checkCudaErrors(cudaEventRecord(s->model->client[s->op->id], s->stream[s->op->branch]));
#else
	checkCudaErrors(cudaEventRecord(s->model->client, s->stream[s->op->branch]));
#endif
	/* Apply gradient to model */
	checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], s->model->elements, &one, (float *)(s->model->gradient->dev), 1, (float *)(s->model->data->dev), 1));

	/* Apply gradient to the parameter server model (using a different handle/stream) */
#ifdef UPDATE_MODEL_INCREMENTALLY
	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client[s->op->id], 0));
#else
	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
#endif
	checkCublasStatus(cublasSaxpy (s->modelSynchronisationHandle, s->model->elements, &one, (float *)(s->model->gradient->dev), 1, (float *)(s->theModel->data->dev), 1));

#ifdef UPDATE_MODEL_INCREMENTALLY
	checkCudaErrors(cudaEventRecord(s->model->server[s->op->id], s->modelSynchronisationStream));
	cudaEventSynchronize (s->model->server[s->op->id]);
#else
	checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
	cudaEventSynchronize (s->model->server);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	/* Return read-write local variables to kernel when the dataflow execution completes */
	crossbowListAppend (s->locals[s->op->id], losses);

	return;
}
