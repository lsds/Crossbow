#include "sma.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUSynchronousModelAveraging (crossbowExecutionContextP ctx, int first, int clock) {

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

#ifdef EAMSGD__NORMALIZE
	int count = 0;
#endif

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
#ifdef EAMSGD__NORMALIZE
			count ++;
#endif

			/* At this point, the replica can be used again for the next task (unless we decide to copy the base model) */
			if (! ctx->modelmanager->replicas[id]->conf->_copy)
				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
		}
	}

	/* Update default model */

#ifdef EAMSGD__NORMALIZE
	invalidConditionException (count > 0);
	float factor = one / (float) count;

	info("Normalise gradient by 1/%d (or %.3f)\n", count, factor);

	checkCublasStatus(cublasSscal (
		defaultDev->modelSynchronisationHandle,
		defaultModel->elements,
		&(factor),
		(float *)(defaultModel->gradient->dev), 1));
#endif

#ifdef EAMSGD__APPLY_MOMENTUM
	/* Apply momentum to default base model gradient */
	if (defaultModel->conf->momentum > 0) {

		dbg("Apply momentum (u is %.3f)\n", defaultModel->conf->momentum);

		checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultModel->elements,
			&(defaultModel->conf->momentum),
			(float *)(defaultModel->last->dev), 1,
			(float *)(defaultModel->gradient->dev), 1));

			/* Copy base model gradient to base model last */
			crossbowDataBufferCopyDeviceRegion
				(defaultModel->last, defaultModel->gradient, defaultDev->modelSynchronisationStream);
	}
#endif

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

	for (id = first; id < ctx->modelmanager->size; ++id) {

		if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == defaultDev->id) {

			if (ctx->modelmanager->replicas[id]->conf->_copy > 0) {

				info("Copy base model to replicas because of learning rate drop\n");
				crossbowDataBufferCopyDeviceRegion (
					ctx->modelmanager->replicas[id]->data, 
					defaultModel->data, 
					defaultDev->modelSynchronisationStream);
				
#ifdef EAMSGD__SHARE_MOMENTUM
				crossbowDataBufferCopyDeviceRegion (
					ctx->modelmanager->replicas[id]->last, 
					defaultModel->last, 
					defaultDev->modelSynchronisationStream);
#endif
				/* Reset signal */
				ctx->modelmanager->replicas[id]->conf->_copy = 0;

				/* Now, release lock on replicas */
				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
			}
		}
	}
}

static void crossbowSynchronisationMultiGPUSynchronousModelAveraging (crossbowExecutionContextP ctx, int first, int clock) {

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
		
		/* Sum all model differences */
		for (id = first; id < ctx->modelmanager->size; ++id) {
			
			if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {
				
				// info("Sum diff. and update replica #%02d (%d updates)\n", id, ctx->modelmanager->replicas[id]->updates);
				
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
				
				/* Modified on 3 Aug. 2018 */
				//float asum1 = 0;
				//checkCublasStatus(cublasSasum (dev->modelSynchronisationHandle, model->elements, (float *) (model->diff->dev), 1, &asum1));
				//float asum2 = 0;
				//checkCublasStatus(cublasSasum (dev->modelSynchronisationHandle, model->elements, (float *) (ctx->modelmanager->replicas[id]->gradient->dev), 1, &asum2));
				//info("Differences for replica %02d: global %10.5f local %10.5f\n", id, asum1, asum2);
				
				//float beta = -1;
				//if (asum1 > asum2) {
				///* Update model replica */
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(minusalpha),
					// &(beta),
					(float *)(model->diff->dev), 1,
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1));
				// }
				
				/* Accumulate difference */
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(alpha),
					(float *)(model->diff->dev), 1,
					(float *)(model->gradient->dev), 1));
#ifdef EAMSGD__NORMALIZE
				count ++;
#endif
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
	

	crossbowSynchronisationAccumulateGradientsAcrossDevices (ctx, defaultModel, defaultDev);
	/* CUDA calls redirected to default device, but reset it anyway */
	checkCudaErrors (cudaSetDevice(defaultDev->id));
	
	/*
	float asum_def = 0;
	checkCublasStatus(cublasSasum (defaultDev->modelSynchronisationHandle, defaultModel->elements, (float *) (defaultModel->gradient->dev), 1, &asum_def));
	info("Average gradient magnitude: %10.5f\n", asum_def);
	*/
	
	/* Update default model */

#ifdef EAMSGD__NORMALIZE
	invalidConditionException (count > 0);
	float factor = one / (float) count;

	info("Normalise gradient by 1/%d (or %.3f)\n", count, factor);

	checkCublasStatus(cublasSscal (
		defaultDev->modelSynchronisationHandle,
		defaultModel->elements,
		&(factor),
		(float *)(defaultModel->gradient->dev), 1));
#endif

#ifdef EAMSGD__APPLY_MOMENTUM
	/* Apply momentum to default base model gradient */
	if (defaultModel->conf->momentum > 0) {
		
		defaultModel->conf->momentum = 0.9;
		dbg("Apply momentum (u is %.3f)\n", defaultModel->conf->momentum);

		checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultModel->elements,
			&(defaultModel->conf->momentum),
			(float *)(defaultModel->last->dev), 1,
			(float *)(defaultModel->gradient->dev), 1));

			/* Copy base model gradient to base model last */
			crossbowDataBufferCopyDeviceRegion
				(defaultModel->last, defaultModel->gradient, defaultDev->modelSynchronisationStream);
	}
#endif
	
	/* Apply default base model gradient */
	
	/* Normalise by clock 
	 * 
	 * float factor = one / (float) clock;
	 * info("Normalise by %.5f (clock is %d)\n", factor, clock);
	 */
	
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
	
	if (copies > 0) {
		info("%d copies; disable models (except one...)\n", copies);
       crossbowModelManagerDisableModels (ctx->modelmanager);
		defaultModel->conf->alpha = 0.1;
    }
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
					
#ifdef EAMSGD__SHARE_MOMENTUM
					crossbowDataBufferCopyDeviceRegion (
						ctx->modelmanager->replicas[id]->last, 
						model->last, 
						dev->modelSynchronisationStream);
#endif
					
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
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUSynchronousModelAveraging (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUSynchronousModelAveraging  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
