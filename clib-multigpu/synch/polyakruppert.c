#include "polyakruppert.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUPolyakRuppert (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

	int id;

	float one = 1;
	float minusone = -one;

	float alpha, minusalpha;
	float scaleFactor = 1. / (float) ctx->modelmanager->size;
	float runningAverageFactor = 1. / (float) (clock + 1);

	crossbowDeviceP defaultDev;
	crossbowModelP defaultModel;

	defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	defaultModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);

	alpha = defaultModel->conf->alpha;
	minusalpha = -alpha;

	dbg("In Beta synchronisation, alpha is %.5f, scale factor is %.5f, clock is %d, running average factor is %.5f\n", alpha, scaleFactor, clock, runningAverageFactor);

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

	/* Sum all models */
	for (id = first; id < ctx->modelmanager->size; ++id) {

		if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == defaultDev->id) {

			/* Accumulate replicas (scaled by 1 / p) */
			checkCublasStatus(cublasSaxpy (
					defaultDev->modelSynchronisationHandle,
					defaultModel->elements,
					&(scaleFactor),
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1,
					(float *)(defaultModel->gradient->dev), 1));

			/* Compute difference from base model (replica->data - base->data) in two steps:
			 *
			 * base->last = replica->data
			 * base->last = base->last - base->data
			 */
			crossbowDataBufferCopyDeviceRegion (defaultModel->last, ctx->modelmanager->replicas[id]->data, defaultDev->modelSynchronisationStream);
			checkCublasStatus(cublasSaxpy (
					defaultDev->modelSynchronisationHandle,
					defaultModel->elements,
					&(minusone),
					(float *)(defaultModel->data->dev), 1,
					(float *)(defaultModel->last->dev), 1));

			/* Update model replica */
			if (alpha != 0) {
				checkCublasStatus(cublasSaxpy (
					defaultDev->modelSynchronisationHandle,
					defaultModel->elements,
					&(minusalpha),
					(float *)(defaultModel->last->dev), 1,
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1));
			}

			/* At this point, the replica can be used again for the next task (unless we decide to copy the base model) */
			if (! ctx->modelmanager->replicas[id]->conf->_copy)
				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));

			/* A hack: change the learning rate...assuming it is fixed */
			/*
			ctx->modelmanager->replicas[id]->conf->learningRate =
			ctx->modelmanager->replicas[id]->conf->learningRate * (float) (pow (clock, -0.5));
			*/
		}
	}

	/* Compute difference in two steps:
	 *
	 * default->last = default->gradient
	 * default->last = default->last - default->data
	 */
	crossbowDataBufferCopyDeviceRegion (defaultModel->last, defaultModel->gradient, defaultDev->modelSynchronisationStream);
	checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultModel->elements,
			&(minusone),
			(float *)(defaultModel->data->dev), 1,
			(float *)(defaultModel->last->dev), 1));

	/* Update default model */
	checkCublasStatus(cublasSaxpy (
		defaultDev->modelSynchronisationHandle,
		defaultModel->elements,
		&(runningAverageFactor),
		(float *)(defaultModel->last->dev), 1,
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
				crossbowDataBufferCopyDeviceRegion (ctx->modelmanager->replicas[id]->data, defaultModel->data, defaultDev->modelSynchronisationStream);
				/* Reset signal */
				ctx->modelmanager->replicas[id]->conf->_copy = 0;

				/* Now, release lock on replicas */
				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
			}
		}
	}
}

static void crossbowSynchronisationMultiGPUPolyakRuppert (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

	int id;

	float one = 1;
	float minusone = -one;

	float alpha, minusalpha;

	float scaleFactor, runningAverageFactor;

	int ndx;

	crossbowDeviceP  dev;
	crossbowModelP model;

	crossbowDeviceP  defaultDev;
	crossbowModelP defaultModel;

	defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	defaultModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);

	alpha = defaultModel->conf->alpha;
	minusalpha = -alpha;

	scaleFactor = 1. / (float) ctx->modelmanager->size;
	runningAverageFactor = 1. / (float) (clock + 1);

	dbg("In Beta synchronisation, alpha is %.5f, scale factor is %.5f, and clock is %d\n", alpha, scaleFactor, clock);

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

		/* Sum all models */
		for (id = first; id < ctx->modelmanager->size; ++id) {

			if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {

				/* Accumulate replicas (scaled by 1 / p) */
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(scaleFactor),
					(float *)(ctx->modelmanager->replicas[id]->data->dev), 1,
					(float *)(model->gradient->dev), 1));

				/* Compute difference (replica->data - base->data) in two steps:
				 *
				 * base->last = replica->data
				 * base->last = base->last - base->data
				 */
				crossbowDataBufferCopyDeviceRegion(model->last, ctx->modelmanager->replicas[id]->data, dev->modelSynchronisationStream);
				checkCublasStatus(cublasSaxpy (
					dev->modelSynchronisationHandle,
					model->elements,
					&(minusone),
					(float *)(model->data->dev), 1,
					(float *)(model->last->dev), 1));

				/* Update model replica */
				if (alpha != 0) {
					checkCublasStatus(cublasSaxpy (
							dev->modelSynchronisationHandle,
							model->elements,
							&(minusalpha),
							(float *)(model->last->dev), 1,
							(float *)(ctx->modelmanager->replicas[id]->data->dev), 1));
				}

				/* At this point, the replica can be used again for the next task (unless we decide to copy the base model) */
				if (! ctx->modelmanager->replicas[id]->conf->_copy)
					checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, dev->modelSynchronisationStream));
			}
        }
#ifdef DEVICE_SYNCHRONIZE
		/* Wait until differences have been accumulated */
		checkCudaErrors(cudaDeviceSynchronize());
#endif
		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
	}

	crossbowSynchronisationAccumulateGradientsAcrossDevices (ctx, defaultModel, defaultDev);
	/* CUDA calls redirected to default device */

	/* Compute difference in two steps:
	 *
	 * default->last = default->gradient
	 * default->last = default->last - default->data
	 */
	crossbowDataBufferCopyDeviceRegion (defaultModel->last, defaultModel->gradient, defaultDev->modelSynchronisationStream);
	checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultModel->elements,
			&(minusone),
			(float *)(defaultModel->data->dev), 1,
			(float *)(defaultModel->last->dev), 1));

	/* Update default model */
	checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultModel->elements,
			&(runningAverageFactor),
			(float *)(defaultModel->last->dev), 1,
			(float *)(defaultModel->data->dev), 1));

	/* Copy default model to all other devices */
	crossbowSynchronisationSynchroniseModelAcrossDevices (ctx, defaultModel, defaultDev, 0);

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

		/* Update model replicas based on new base model */

		for (id = first; id < ctx->modelmanager->size; ++id) {

			if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {

				if (ctx->modelmanager->replicas[id]->conf->_copy > 0) {

					info("Copy base model to replicas because of learning rate drop\n");
					crossbowDataBufferCopyDeviceRegion (ctx->modelmanager->replicas[id]->data, model->data, dev->modelSynchronisationStream);
					/* Reset signal */
					ctx->modelmanager->replicas[id]->conf->_copy = 0;

					checkCudaErrors(cudaEventRecord (ctx->modelmanager->replicas[id]->updated, dev->modelSynchronisationStream));
				}
			}
		}
	}

	return;
}

void crossbowSynchronisationPolyakRuppert (crossbowExecutionContextP ctx, int first, int clock) {
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUPolyakRuppert (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUPolyakRuppert  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
