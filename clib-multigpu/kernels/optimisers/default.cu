#include "default.h"

void crossbowKernelOptimiserDefault (crossbowStreamP s) {

	crossbowDataBufferP model, gradient, last, theModel;
	float rate;

	int elements;

	/* Constants */
	float one = 1;
	float minusone = -1;

	/* Get model replica data buffer */
	model = s->model->data;

	/* Get current and gradient data buffer */
	gradient = s->model->gradient;
	last = s->model->last;

	/* Get base model data buffer */
	theModel = s->theModel->data;

	elements = s->model->elements;

	/* Apply weight decay */
	if (s->model->conf->weightDecay > 0)
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(s->model->conf->weightDecay), (float *) (model->dev), 1, (float *) (gradient->dev), 1));

	rate = minusone * crossbowSolverConfGetLearningRate(s->model->conf, s->task);

	if (s->model->conf->momentum > 0) {

		if (s->model->conf->momentumMethod == NESTEROV) {
			err("Nesterov's momentum has been disabled\n");
		}

		/* Scale gradient based on learning rate */
		checkCublasStatus(cublasSscal(s->cublasHandle[s->op->branch], elements, &(rate), (float *) (gradient->dev), 1));

		/* Apply momentum to gradient */
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(s->model->conf->momentum), (float *) (last->dev), 1, (float *) (gradient->dev), 1));

		/* Record event that gradient is ready to be used by parameter server */
		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));

		/* Copy current gradient into last */
		checkCudaErrors(cudaMemcpyAsync(last->dev, gradient->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));

		/* Apply gradient to local model */
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(one), (float *) (gradient->dev), 1, (float *) (model->dev), 1));

		/* Apply gradient to parameter server model (base model) */
		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));

		checkCublasStatus(cublasSaxpy (s->modelSynchronisationHandle, elements, &(one), (float *) (gradient->dev), 1, (float *) (theModel->dev), 1));

		/* Record event that gradient has been applied to base model */
		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
	}
	else {
		/* Record event that gradient is ready to be used by parameter server */
		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));

		/* Apply gradient to local model */
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(rate), (float *) (gradient->dev), 1, (float *) (model->dev), 1));

		/* Apply gradient to parameter server model (base model) */
		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));

		checkCublasStatus(cublasSaxpy (s->modelSynchronisationHandle, elements, &(rate), (float *) (gradient->dev), 1, (float *) (theModel->dev), 1));

		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
	}

	return;
}