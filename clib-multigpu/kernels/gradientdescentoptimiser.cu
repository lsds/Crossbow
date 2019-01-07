#include "gradientdescentoptimiser.h"

#ifndef UPDATE_MODEL_INCREMENTALLY

static void crossbowKernelGradientDescentOptimiserEamsgdModelUpdate  (crossbowStreamP s) {
	(void) s;
	err ("Asynchronous elastic averaging SGD is not supported yet");
	return;
}

static void crossbowKernelGradientDescentOptimiserDownpourModelUpdate (crossbowStreamP s) {
	(void) s;
	err ("Down-pour SGD is not supported yet");
	return;
}

static void crossbowKernelGradientDescentOptimiserDefaultModelUpdate (crossbowStreamP s) {

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

static void crossbowKernelGradientDescentOptimiserWorkerModelUpdate (crossbowStreamP s) {
	
	float minusone = -1;

	float rate;

	int elements = s->model->elements;

	/* Get replica model data buffer */
	crossbowDataBufferP model = s->model->data;

	/* Get replica model gradient data buffer */
	crossbowDataBufferP gradient = s->model->gradient;

	/* Get base model gradient data buffer */
	crossbowDataBufferP theGradient = s->theModel->gradient;

	/* Apply weight decay to gradient, if configured */
	if (s->model->conf->weightDecay > 0)
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], elements, &(s->model->conf->weightDecay), (float *) (model->dev), 1, (float *) (gradient->dev), 1));
	
	/*
	checkCudaErrors(cudaDeviceSynchronize());
	
	float checksum = crossbowDataBufferComputeCheckSum(gradient, 0, s->model->bytes);
	info("Gradient checksum of task %d is %.5f\n", s->task, checksum);
	*/
	
	/* Record event that replica model gradient is ready to be used by the parameter server */
	checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));

	/* Accumulate replica model gradient to base model gradient, applying learning rate */
	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));

	if (s->model->conf->momentumMethod == NESTEROV) {
		err("Nesterov's momentum has been disabled\n");
	}
	rate = minusone * crossbowSolverConfGetLearningRate (s->model->conf, s->task);
	// info("Learning rate is %.5f\n", rate);
	checkCublasStatus(cublasSaxpy (s->modelSynchronisationHandle, elements, &(rate), (float *) (gradient->dev), 1, (float *) (theGradient->dev), 1));

	/* Record event that replica model gradient is not longer required by the parameter server */
	checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));

	return;
}

static void crossbowKernelGradientDescentOptimiserSynchronousEamsgdModelUpdate (crossbowStreamP s) {

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

#endif /* End of function definitions */

void crossbowKernelGradientDescentOptimiser (void *args) {

#ifdef UPDATE_MODEL_INCREMENTALLY
	/* Do nothing */
 	(void) args;
#else
	crossbowStreamP s = (crossbowStreamP) args;

	switch (s->model->type) {

	case DEFAULT:
		dbg("Update replica using default model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case  MULTI_GPU:
			crossbowKernelGradientDescentOptimiserDefaultModelUpdate (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case WORKER:
		dbg("Update replica using worker model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case  MULTI_GPU:
			crossbowKernelGradientDescentOptimiserWorkerModelUpdate  (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case EAMSGD:
		dbg("Update replica using synchronous EAMSGD model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelGradientDescentOptimiserEamsgdModelUpdate  (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case SYNCHRONOUSEAMSGD:
		dbg("Update replica using synchronous EAMSGD model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelGradientDescentOptimiserSynchronousEamsgdModelUpdate  (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case DOWNPOUR:
		dbg("Update replica using (synchronous) DOWNPOUR model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelGradientDescentOptimiserDownpourModelUpdate  (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	default:
		err("Invalid model update type\n");
	}

#endif /* UPDATE_MODEL_INCREMENTALLY */

	return;
}
