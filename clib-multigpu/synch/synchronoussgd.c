#include "synchronoussgd.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUSynchronousSGD (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

	float one = 1;

	crossbowDeviceP defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	crossbowModelP defaultBaseModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);

	/* Redirect all CUDA calls to the default device */
	checkCudaErrors (cudaSetDevice(defaultDev->id));

	/*
	 * Gradients from all replicas are being accumulated to base model gradient buffer
	 * on `defaultDev->modelSynchronisationStream`.
	 */

	/* checkCudaErrors(cudaDeviceSynchronize()); */

	/* Scale gradient */
	float ratio = 1.0 / (float) defaultBaseModel->wpc;

	if (ratio < 1)
		checkCublasStatus(cublasSscal(
				defaultDev->modelSynchronisationHandle,
				defaultBaseModel->elements,
				&(ratio),
				(float *)(defaultBaseModel->gradient->dev),
				1));

	/* Apply momentum to base model gradient */
	if (defaultBaseModel->conf->momentum > 0) {

		checkCublasStatus(cublasSaxpy (
				defaultDev->modelSynchronisationHandle,
				defaultBaseModel->elements,
				&(defaultBaseModel->conf->momentum),
				(float *)(defaultBaseModel->last->dev), 1,
				(float *)(defaultBaseModel->gradient->dev), 1));

		/* Copy base model gradient to base model last */
		crossbowDataBufferCopyDeviceRegion
			(defaultBaseModel->last, defaultBaseModel->gradient, defaultDev->modelSynchronisationStream);
	}

	/* Apply base model gradient to base model */
	checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultBaseModel->elements,
			&(one),
			(float *)(defaultBaseModel->gradient->dev), 1,
			(float *)(defaultBaseModel->data->dev), 1));

	/* Clear previous state */
	checkCudaErrors(cudaMemsetAsync (defaultBaseModel->gradient->dev, 0, defaultBaseModel->bytes, defaultDev->modelSynchronisationStream));

	/* Synchronise replicas on device */
	crossbowSynchronisationSynchroniseModelOnDevice (ctx, first, defaultBaseModel, defaultDev);

	/* checkCudaErrors(cudaDeviceSynchronize()); */

	return;
}

static void crossbowSynchronisationMultiGPUSynchronousSGD (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

	float one = 1;

	int ndx;
	crossbowDeviceP dev;
	crossbowModelP model;

	crossbowDeviceP defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	crossbowModelP defaultBaseModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);

	/* First iterate over all devices record 'accumulated' event */

	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {

		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if ((! crossbowDeviceSelected(dev)) || (dev->id == defaultDev->id)) /* Skip default device */
			continue;

		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));
		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
	}

	crossbowSynchronisationAccumulateGradientsAcrossDevices (ctx, defaultBaseModel, defaultDev);

	/* CUDA calls already redirected to default device. */

	/* Scale gradient */
	float ratio = 1.0 / (float) defaultBaseModel->wpc;

	checkCublasStatus(cublasSscal (
			defaultDev->modelSynchronisationHandle,
			defaultBaseModel->elements,
			&(ratio),
			(float *)(defaultBaseModel->gradient->dev), 1));

	/* Apply momentum to base model gradient */
	if (defaultBaseModel->conf->momentum > 0) {

		checkCublasStatus(cublasSaxpy (
				defaultDev->modelSynchronisationHandle,
				defaultBaseModel->elements,
				&(defaultBaseModel->conf->momentum),
				(float *)(defaultBaseModel->last->dev), 1,
				(float *)(defaultBaseModel->gradient->dev), 1));

		/* Copy base model gradient to base model last */
		crossbowDataBufferCopyDeviceRegion
			(defaultBaseModel->last, defaultBaseModel->gradient, defaultDev->modelSynchronisationStream);
	}

	/* Apply base model gradient */
	checkCublasStatus(cublasSaxpy (
			defaultDev->modelSynchronisationHandle,
			defaultBaseModel->elements,
			&(one),
			(float *)(defaultBaseModel->gradient->dev), 1,
			(float *)(defaultBaseModel->data->dev), 1));

	crossbowSynchronisationSynchroniseModelAcrossDevices (ctx, defaultBaseModel, defaultDev, 0);

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
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUSynchronousSGD (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUSynchronousSGD  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
