#include "synchronouseamsgd.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUSynchronousElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

	int id;

	float one = 1;
	float minusone = -one;

	float alpha, minusalpha;

	crossbowDeviceP defaultDev;
	crossbowModelP defaultModel;

	defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	defaultModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);

	alpha = defaultModel->conf->alpha;
	minusalpha = -alpha;

#ifdef GPU_VERBOSE
	dbg("%d visible updates (across all replicas)\n", crossbowModelManagerNumberOfVisibleUpdates (ctx->modelmanager));
#endif

	/* Redirect all CUDA calls to the default device */
	checkCudaErrors (cudaSetDevice(defaultDev->id));

#ifdef DEVICE_SYNCHRONIZE
	/* Wait until local models have been updated */
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	/* Wait until base model has been updated (from a previous iteration)
	 * since we are using it to compute its difference from replicas.
	 */
	checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, defaultModel->updated, 0));

	/* Reset accumulated difference (stored in base model gradient buffer) */
	checkCudaErrors(cudaMemsetAsync(defaultModel->gradient->dev, 0, defaultModel->bytes, defaultDev->modelSynchronisationStream));

	/* Sum all model differences */
	for (id = first; id < ctx->modelmanager->size; ++id) {

		if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == defaultDev->id) {

			/* Compute difference from base model (replica->data - base->data) in two steps:
			 *
			 * base->diff = replica->data
			 * base->diff = base->diff - base->data
			 */
			crossbowDataBufferCopyDeviceRegion (defaultModel->diff, ctx->modelmanager->replicas[id]->data, defaultDev->modelSynchronisationStream);
			checkCublasStatus(cublasSaxpy (
					defaultDev->modelSynchronisationHandle,
					defaultModel->elements,
					&(minusone),
					(float *)(defaultModel->data->dev), 1,
					(float *)(defaultModel->diff->dev), 1));

			/* Update model replica */
			checkCublasStatus(cublasSaxpy (
					defaultDev->modelSynchronisationHandle,
					defaultModel->elements,
					&(minusalpha),
					(float *)(defaultModel->diff->dev), 1,
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1));

			/* Accumulate difference */
			checkCublasStatus(cublasSaxpy (
					defaultDev->modelSynchronisationHandle,
					defaultModel->elements,
					&(alpha),
					(float *)(defaultModel->diff->dev), 1,
					(float *)(defaultModel->gradient->dev), 1));

			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
		}
	}

	/* Apply default base model gradient */

	checkCublasStatus(cublasSaxpy (
		defaultDev->modelSynchronisationHandle,
		defaultModel->elements,
		&(one),
		(float *)(defaultModel->gradient->dev), 1,
		(float *)(defaultModel->data->dev), 1));

	/* Record event that the base model has been updated */
	checkCudaErrors(cudaEventRecord(defaultModel->updated, defaultDev->modelSynchronisationStream));

#ifdef DEVICE_SYNCHRONIZE
	/* Wait until default model has been updated */
	checkCudaErrors(cudaDeviceSynchronize());
#endif
	
	return;
}

/*
 * Without hierarchy...
 */
static void crossbowSynchronisationMultiGPUSynchronousElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

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

#ifdef EAMSGD__NORMALIZE
	int count = 0;
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
		
		/* Compute model differences and accumulate them on default device */
		for (id = first; id < ctx->modelmanager->size; ++id) {
			
			if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {
				
				
				checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, defaultModel->accumulated, 0));
				
				/* Compute difference (replica->data - base->data) in two steps:
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
				
				/* Record event that the diff has been computed */
				
				checkCudaErrors(cudaEventRecord (ctx->modelmanager->replicas[id]->accumulated, dev->modelSynchronisationStream));
				
				/* Update model replica */
				
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(minusalpha),
					(float *)(model->diff->dev), 1,
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1));
				
				/* Accumulate difference (but on the default device) */
				/*
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(alpha),
					(float *)(model->diff->dev), 1,
					(float *)(model->gradient->dev), 1));
				*/
				
				// if (dev->id != defaultDev->id) {
				
				checkCudaErrors (cudaSetDevice(defaultDev->id));
				
				checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, ctx->modelmanager->replicas[id]->accumulated, 0));
				
				cudaMemcpyPeerAsync (defaultModel->temp->dev, defaultDev->id, model->diff->dev, dev->id, defaultModel->bytes, defaultDev->modelSynchronisationStream);
				
				checkCublasStatus(cublasSaxpy (
                	defaultDev->modelSynchronisationHandle,
                	defaultModel->elements,
                	&(alpha),
                	(float *)(defaultModel->temp->dev), 1,
                	(float *)(defaultModel->gradient->dev), 1));
				
				checkCudaErrors(cudaEventRecord (defaultModel->accumulated, defaultDev->modelSynchronisationStream));
				
				checkCudaErrors (cudaSetDevice(dev->id));
				
				// }
				
#ifdef EAMSGD__NORMALIZE
				count ++;
#endif
				/* 
				 * At this point, the replica can be used again for the next task 
				 * (unless we decide to copy the base model) 
				 */
				
				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, 
					dev->modelSynchronisationStream));
			}
        }
#ifdef DEVICE_SYNCHRONIZE
		/* Wait until differences have been accumulated */
		checkCudaErrors(cudaDeviceSynchronize());
#endif
		
		/* No need for this event... */
		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
	}
	
	/* crossbowExecutionContextAccumulateGradientsAcrossDevices (ctx, defaultModel, defaultDev); */
	
	/* CUDA calls redirected to default device, but reset it anyway */
	checkCudaErrors (cudaSetDevice(defaultDev->id));
	
	/* Update default model */

	/* Apply default base model gradient */
	
	checkCublasStatus(cublasSaxpy (
		defaultDev->modelSynchronisationHandle,
		defaultModel->elements,
		&(one), /* &(factor) */
		(float *)(defaultModel->gradient->dev), 1,
		(float *)(defaultModel->data->dev), 1));

	/* Copy default model to all other devices */
	crossbowSynchronisationSynchroniseModelAcrossDevices (ctx, defaultModel, defaultDev, copies);

#ifdef DEVICE_SYNCHRONIZE
	/* Wait until all device models have been updated */
	checkCudaErrors(cudaDeviceSynchronize());
#endif
	
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

	}

	return;
}

void crossbowSynchronisationSynchronousElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUSynchronousElasticAveragingSGD (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUSynchronousElasticAveragingSGD  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
