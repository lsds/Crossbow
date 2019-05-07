#include "common.h"

void crossbowSynchronisationAllReduceGradientsAcrossDevices (crossbowExecutionContextP ctx) {
	int ndx = 0;
	crossbowDeviceP dev = NULL;
	crossbowModelP model = NULL;
#ifndef USE_NCCL
	(void) ndx;
	(void) dev;
	(void) model;
	(void) ctx;
	err("NCCL is disabled");
#else
	checkNcclErrors(ncclGroupStart());
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		
		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;
		
		/* Redirect all CUDA calls to the current device */
		checkCudaErrors (cudaSetDevice(dev->id));
		
		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		/*
		 * We are about to use dev->modelSynchronisationStream.
		 * A. What events do we have to wait?
		 *
		 * Obviously, we have to wait until all differences are
		 * accumulated.
		 *
		 * B. What are the send and receive buffers?
		 *
		 * The send buffer for each device are (base-)model->gradient.
		 * The receive buffers are (base-)model->diff.
		 */
		
		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, model->accumulated, 0));
		
		checkCudaErrors(cudaMemsetAsync(model->diff->dev, 0, model->bytes, dev->modelSynchronisationStream));
		
		checkNcclErrors(ncclAllReduce(
			model->gradient->dev,
			model->diff->dev, 
			model->elements,
			ncclFloat, 
			ncclSum, 
			ctx->comms[dev->id], 
			dev->modelSynchronisationStream));
    }
    checkNcclErrors(ncclGroupEnd());
#endif
	return;
}

void crossbowSynchronisationAccumulateGradientsAcrossDevices (crossbowExecutionContextP ctx, crossbowModelP defaultModel, crossbowDeviceP defaultDev) {
	int ndx;
	crossbowDeviceP dev;
	crossbowModelP model;
	float one = 1;
#ifndef USE_NCCL
	/* Base model gradients are accumulated on the default device. */
	
	/* Redirect all CUDA calls to the default device (the master) */
	checkCudaErrors (cudaSetDevice(defaultDev->id));

	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {

		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		/* Skip default device */
		if ((! crossbowDeviceSelected(dev)) || (dev->id == defaultDev->id)) 
			continue;
		
		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		/* Wait until partial gradients have been accumulated on current device */
		checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, model->accumulated, 0));

		/* Fetch accumulated gradient from current device */
		cudaMemcpyPeerAsync (defaultModel->temp->dev, defaultDev->id, model->gradient->dev, dev->id, 
			defaultModel->bytes, defaultDev->modelSynchronisationStream);

		/* Accumulate gradient at master */
		checkCublasStatus(cublasSaxpy (
				defaultDev->modelSynchronisationHandle,
				defaultModel->elements,
				&(one),
				(float *)(defaultModel->temp->dev), 1,
				(float *)(defaultModel->gradient->dev), 1));
	}
#else

	(void) one;

	checkNcclErrors(ncclGroupStart());
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {

		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;

		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

		checkNcclErrors(ncclReduce(model->gradient->dev, defaultModel->gradient->dev, model->bytes, 
            ncclChar, ncclSum, defaultDev->id, ctx->comms[dev->id], dev->modelSynchronisationStream));
	}
	checkNcclErrors(ncclGroupEnd());
#endif
	return;
}

void crossbowSynchronisationSynchroniseModelAcrossDevices (crossbowExecutionContextP ctx, 
	crossbowModelP defaultModel, crossbowDeviceP defaultDev, unsigned shareMomentum) {
	int ndx;
	crossbowDeviceP dev;
	crossbowModelP model;

	/* Assumes that CUDA calls are already redirected to default device */
#ifndef USE_NCCL
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {

		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;

		if (dev->id != defaultDev->id) {

			/* Get base model for current device */
			model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

			/* Copy default device's base model */
			cudaMemcpyPeerAsync (model->data->dev, dev->id, defaultModel->data->dev, 
                defaultDev->id, model->bytes, defaultDev->modelSynchronisationStream);

#ifdef EAMSGD__SHARE_MOMENTUM
			if (shareMomentum) {
            	dbg("Share momentum\n");
				cudaMemcpyPeerAsync (model->last->dev, dev->id, defaultModel->last->dev, 
                	defaultDev->id, model->bytes, defaultDev->modelSynchronisationStream);
			}
#else
			(void) shareMomentum;
#endif
		}

		/* Record multi-GPU synchronisation event */
		checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], 
            defaultDev->modelSynchronisationStream));
	}
#else

	(void) defaultModel;
	(void) shareMomentum;

	checkNcclErrors(ncclGroupStart());
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
	
		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;

		/* Get base model for current device */
		model = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);
		
		checkCudaErrors (cudaSetDevice(dev->id));

		checkNcclErrors(ncclBcast(model->data->dev, model->bytes, ncclChar, defaultDev->id, 
			ctx->comms[dev->id], dev->modelSynchronisationStream));
	}
	checkNcclErrors(ncclGroupEnd());
	
	checkCudaErrors (cudaSetDevice(defaultDev->id));
	
	/* Record multi-GPU synchronisation event */
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		
		/* Get current device */
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (! crossbowDeviceSelected(dev))
			continue;
		
		checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], 
			dev->modelSynchronisationStream));
	}
#endif
	return;
}

void crossbowSynchronisationSynchroniseModelOnDevice (crossbowExecutionContextP ctx, int first, crossbowModelP model, crossbowDeviceP dev) {

	int id;

	/* Copy `model` to all replicas on device `dev`. Assumes that all CUDA calls have been redirected to that device. */
	
	/* Wait until device's base model is synchronised */
	checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, ctx->modelmanager->synched[dev->id], 0));
	
	for (id = first; id < ctx->modelmanager->size; ++id) {

		if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {

			dbg("Copy base model to model replica #%d\n", id);
			crossbowDataBufferCopyDeviceRegion
				(ctx->modelmanager->replicas[id]->data, model->data, dev->modelSynchronisationStream);

			/* Record event that replica has been updated. */
			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, dev->modelSynchronisationStream));
		}
	}
	return;
}
