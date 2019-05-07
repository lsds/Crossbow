#include "synchronoussgd.h"

void crossbowKernelOptimiserSynchronousSGD (crossbowStreamP s) {
	
	float minusone = -1;
	
	/* Number of model (and gradient) parameters */
	int elements = s->model->elements;

	/* Get replica model data buffer */
	crossbowDataBufferP model = s->model->data;

	/* Get replica's gradient data buffer */
	crossbowDataBufferP gradient = s->model->gradient;

	/* Get base model's gradient data buffer */
	crossbowDataBufferP theGradient = s->theModel->gradient;
	
	/* Apply weight decay to gradient, if set */
	if (s->model->conf->weightDecay > 0) {
		checkCublasStatus(cublasSaxpy (s->cublasHandle[s->op->branch], 
			elements, 
			&(s->model->conf->weightDecay), 
			(float *) (model->dev), 1, 
			(float *) (gradient->dev), 1));
	}
	
	/* For debugging purposes
	 * 
	 * checkCudaErrors(cudaDeviceSynchronize());
	 * 
	 * float checksum = crossbowDataBufferComputeCheckSum(gradient, 0, s->model->bytes);
	 * info("Gradient checksum of task %d is %.5f\n", s->task, checksum);
	 */
	
	/* Record event that gradient is ready to be used */
	checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
	
	/* Accumulate replica's gradient to base model's gradient, applying learning rate */
	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
	
	if (s->model->conf->momentumMethod == NESTEROV) {
		err("Nesterov's momentum has been disabled\n");
	}
	
	float rate = minusone * crossbowSolverConfGetLearningRate (s->model->conf, s->task);
	checkCublasStatus(cublasSaxpy (s->modelSynchronisationHandle, 
		elements, 
		&(rate), 
		(float *) (gradient->dev), 1, 
		(float *) (theGradient->dev), 1));

	/* Record event that replica model gradient is no longer required */
	checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));

	return;
}
