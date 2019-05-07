#include "sma.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUSynchronousModelAveraging (crossbowExecutionContextP ctx, int first) {

	(void) ctx;
	(void) first;
	err("Temporarily disabled");
	return;
}

static void crossbowSynchronisationMultiGPUSynchronousModelAveraging (crossbowExecutionContextP ctx, int first) {

	int id;
	
	float one = 1;
	float minusone = -one;
	
	float alpha, minusalpha;
	
	int ndx;
	
	crossbowDeviceP  dev;
	crossbowModelP model;
	
	crossbowDeviceP  defaultDev;
	crossbowModelP defaultModel;
	
	defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	defaultModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);
	
	alpha = defaultModel->conf->alpha;
	minusalpha = -alpha;

#ifdef GPU_VERBOSE
	dbg("%d visible updates (across all replicas)\n", crossbowModelManagerNumberOfVisibleUpdates (ctx->modelmanager));
#endif

	unsigned copies = 0;
	
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {

		/* Current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev)) /* Do not skip default device */
			continue;
		
		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));

#ifdef DEVICE_SYNCHRONIZE
		/* Wait until local models have been updated */
		checkCudaErrors(cudaDeviceSynchronize());
#endif
		
		/* Wait until base model has been updated (from a previous iteration)
		 * since we are using it to compute its difference from replicas.
		 */
		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, model->updated, 0));
		
		/* Reset accumulated difference (stored in base model gradient buffer) */
		checkCudaErrors(cudaMemsetAsync (model->gradient->dev, 0, model->bytes, dev->modelSynchronisationStream));
		
		/* Sum all model differences */
		for (id = first; id < ctx->modelmanager->size; ++id) {
			
			if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {
				
				/* 
				 * Compute difference (replica->data - base->data) in two steps:
				 *
				 * base->diff = replica->data
				 * base->diff = base->diff - base->data
				 */
				crossbowDataBufferCopyDeviceRegion(
					model->diff, 
					/* ctx->modelmanager->replicas[id]->data, */
					ctx->modelmanager->replicas[id]->diff,
					dev->modelSynchronisationStream);
				
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(minusone),
					(float *)(model->data->dev), 1,
					(float *)(model->diff->dev), 1));
				
				/* Update model replica */
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(minusalpha),
					// &(beta),
					(float *)(model->diff->dev), 1,
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1));
				
				/* Accumulate difference */
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(alpha),
					(float *)(model->diff->dev), 1,
					(float *)(model->gradient->dev), 1));
				
				/* 
				 * At this point, the replica can be used again for the next task 
				 * (unless we decide to copy the base model) 
				 */
				
				if (! ctx->modelmanager->replicas[id]->conf->_copy) {
					checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, 
						dev->modelSynchronisationStream));
				} else {
					/* How many replicas are going to be synched with the default base model? */
					copies ++;
				}
			}
        }
#ifdef DEVICE_SYNCHRONIZE
		/* Wait until differences have been accumulated */
		checkCudaErrors(cudaDeviceSynchronize());
#endif
		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
	}
	
	crossbowSynchronisationAllReduceGradientsAcrossDevices (ctx);
	
	/* Redirected CUDA calls to default device */
	checkCudaErrors (cudaSetDevice(defaultDev->id));
	
	/* Update base model on each device */
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		/* Current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev)) /* Do not skip default device */
			continue;
		
		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));
	
#ifdef EAMSGD__APPLY_MOMENTUM
		/* Apply momentum to default base model gradient */
		if (model->conf->momentum > 0) {
			/* TODO Remove hard-coded value */
			model->conf->momentum = 0.9;
			dbg("Apply momentum (u is %.3f)\n", defaultModel->conf->momentum);

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
#endif
		
		/* Apply accumulated and reduced differences to base model */
		checkCublasStatus(cublasSaxpy (
			dev->modelSynchronisationHandle,
			model->elements,
			&(one),
			(float *)(model->diff->dev), 1,
			(float *)(model->data->dev), 1));
		
		/* Record multi-GPU synchronisation event */
                checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], dev->modelSynchronisationStream));
	}

#ifdef DEVICE_SYNCHRONIZE
	/* Wait until all device models have been updated */
	checkCudaErrors(cudaDeviceSynchronize());
#endif
	
	int copyall = (copies > 0) ? 1 : 0;

	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {

		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;

		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));

		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

		/* Wait until base models are synchronised */
		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, ctx->modelmanager->synched[dev->id], 0));
		
		/* Record event that the base model has been updated */
		checkCudaErrors(cudaEventRecord(model->updated, dev->modelSynchronisationStream));

		/* Update model replicas based on new base model */

		for (id = first; id < ctx->modelmanager->size; ++id) {

			if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {
				
				// if (ctx->modelmanager->replicas[id]->conf->_copy > 0) {
				if (copyall > 0) {
					info("Copy base model to replicas because of learning rate drop\n");
					crossbowDataBufferCopyDeviceRegion (
						ctx->modelmanager->replicas[id]->data, 
						model->data, 
						dev->modelSynchronisationStream);
					
					/* Reset signal */
					ctx->modelmanager->replicas[id]->conf->_copy = 0;

					checkCudaErrors(cudaEventRecord (ctx->modelmanager->replicas[id]->updated, 
						dev->modelSynchronisationStream));
				}
			}
		}
	}

	return;
}

void crossbowSynchronisationSMA (crossbowExecutionContextP ctx, int first, int clock) {
	
	(void) clock;
	
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUSynchronousModelAveraging (ctx, first);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUSynchronousModelAveraging  (ctx, first);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
