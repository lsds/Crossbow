#include "default.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUDefault (crossbowExecutionContextP ctx, int first) {
	int id;
	
	/* Get the base model for the default device */
	crossbowDeviceP dev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	crossbowModelP theModel = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

	/* Redirect all CUDA calls to the default device */
	checkCudaErrors (cudaSetDevice(dev->id));

	/* checkCudaErrors(cudaDeviceSynchronize()); */

	/* Copy `theModel` to all device replicas */
	for (id = first; id < ctx->modelmanager->size; ++id) {

		if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == dev->id) {
			
			/* Wait until all partial gradients have been applied to the base model */
			checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, 
				ctx->modelmanager->replicas[id]->server, 0));

			dbg("Copy the base model to model replica #%d\n", id);
			crossbowDataBufferCopyDeviceRegion (
				ctx->modelmanager->replicas[id]->data, 
				theModel->data, 
				dev->modelSynchronisationStream);

			/* 
			 * Record event. This ought to remove the need to synchronise 
			 * on `ctx->modelSynchronisationStream` 
			 */
			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, 
				dev->modelSynchronisationStream));
		}
	}
	
	/* checkCudaErrors(cudaDeviceSynchronize()); */
	
	return;
}

static void crossbowSynchronisationMultiGPUDefault (crossbowExecutionContextP ctx, int first) {

	(void)   ctx;
	(void) first;
	err ("Multi-GPU default SGD model synchronisation is not supported yet\n");
	return;
}

void crossbowSynchronisationDefault (crossbowExecutionContextP ctx, int first, int clock) {
	
	(void) clock;
	
	switch (ctx->mode) {
		case SINGLE_GPU: 
			crossbowSynchronisationSingleGPUDefault (ctx, first); 
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUDefault (ctx, first); 
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
