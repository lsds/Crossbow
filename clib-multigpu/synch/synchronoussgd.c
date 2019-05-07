#include "synchronoussgd.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUSynchronousSGD (crossbowExecutionContextP ctx, int first) {
	
	(void) ctx;
	(void) first;
	err("Temporarily disabled\n");
	return;
}

static void crossbowSynchronisationMultiGPUSynchronousSGD (crossbowExecutionContextP ctx, int first) {
	
	float one = 1;
	
	int ndx;
	crossbowDeviceP dev;
	crossbowModelP model;
	
	crossbowDeviceP defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	crossbowModelP defaultBaseModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);
	
	/* First iterate over all devices and record 'accumulated' event */
	
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
	
		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev)) 
			continue;
		
		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));
		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
	}
	
	crossbowSynchronisationAllReduceGradientsAcrossDevices (ctx);
	
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;
		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));

		/* Scale gradient */
		float ratio = 1.0 / (float) defaultBaseModel->wpc;
	
		checkCublasStatus(cublasSscal (
			dev->modelSynchronisationHandle,
			model->elements,
			&(ratio),
			(float *)(model->diff->dev), 1));
		
		/* Apply momentum to base model gradient */
		if (model->conf->momentum > 0) {

			checkCublasStatus(cublasSaxpy (
				dev->modelSynchronisationHandle,
				model->elements,
				&(model->conf->momentum),
				(float *)(model->last->dev), 1,
				(float *)(model->diff->dev), 1));
		
			/* Copy base model gradient to base model last */
			crossbowDataBufferCopyDeviceRegion
				(model->last, model->diff, dev->modelSynchronisationStream);
		}

		/* Apply base model gradient */
		checkCublasStatus(cublasSaxpy (
			dev->modelSynchronisationHandle,
			model->elements,
			&(one),
			(float *)(model->diff->dev), 1,
			(float *)(model->data->dev), 1));
		
		checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], dev->modelSynchronisationStream));
	}
	
	/* Finally, iterate over devices and update model replicas therein (incl. default device) */
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;

		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));

		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

		/* Reset gradient buffer */
		checkCudaErrors(cudaMemsetAsync (model->gradient->dev, 0, model->bytes, dev->modelSynchronisationStream));

		/* Synchronise replicas on device */
		crossbowSynchronisationSynchroniseModelOnDevice (ctx, first, model, dev);
	}

	return;
}

void crossbowSynchronisationSynchronousSGD (crossbowExecutionContextP ctx, int first, int clock) {
	
	(void) clock;
	
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUSynchronousSGD (ctx, first);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUSynchronousSGD  (ctx, first);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
