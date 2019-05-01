#include "synchronouseasgd.h"

void crossbowKernelOptimiserSynchronousElasticAveragingSGD (crossbowStreamP s) {

	crossbowDataBufferP model, gradient, last;
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

	elements = s->model->elements;

	/* Apply weight decay */
	if (s->model->conf->weightDecay > 0)
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(s->model->conf->weightDecay), (float *) (model->dev), 1, (float *) (gradient->dev), 1));

	/*
	checkCudaErrors(cudaDeviceSynchronize());

	float checksum = crossbowDataBufferComputeCheckSum(gradient, 0, s->model->bytes);
	info("Gradient checksum of task %d is %.5f\n", s->task, checksum);
	*/

	rate = minusone * crossbowSolverConfGetLearningRate(s->model->conf, s->task);

	if (s->model->conf->momentum > 0) {

		if (s->model->conf->momentumMethod == NESTEROV) {
			err("Nesterov's momentum has been disabled\n");
		}

		/* Scale gradient based on learning rate */
		checkCublasStatus(cublasSscal(s->cublasHandle[s->op->branch], elements, &(rate), (float *) (gradient->dev), 1));

		/* Apply momentum to gradient */
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(s->model->conf->momentum), (float *) (last->dev), 1, (float *) (gradient->dev), 1));

		/* Copy current gradient buffer into last */
		checkCudaErrors(cudaMemcpyAsync(last->dev, gradient->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
		
		/* Modified on 3 Aug. 2018: Copy existing model into `diff` buffer before updating it */
		checkCudaErrors(cudaMemcpyAsync(s->model->diff->dev, model->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
		
		/* Apply last to local model */
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(one), (float *) (gradient->dev), 1, (float *) (model->dev), 1));
		
		/*
		 * Override semantics of `client` event: now records that the model itself, not the gradient,
		 * is ready to be used by the parameter server.
		 */
		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
	}
	else {
		
		/* Modified on 3 Aug. 2018: Copy existing model into `diff` buffer before updating it */
		checkCudaErrors(cudaMemcpyAsync(s->model->diff->dev, model->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));

		/* Apply gradient to local model */
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(rate), (float *) (gradient->dev), 1, (float *) (model->dev), 1));

		/*
		 * Override semantics of `client` event: now records that the model itself, not the gradient,
		 * is ready to be used by the parameter server.
		 */
		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
	}
}