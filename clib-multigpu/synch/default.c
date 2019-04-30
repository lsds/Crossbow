#include "default.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUDefault (crossbowExecutionContextP ctx, int first, int clock) {

	(void) clock;

	int id;

	crossbowDeviceP defaultDev = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	crossbowModelP defaultBaseModel = crossbowArrayListGet (ctx->modelmanager->baseModels, defaultDev->id);

	/* Redirect all CUDA calls to the default device */
	checkCudaErrors (cudaSetDevice(defaultDev->id));

	/* checkCudaErrors(cudaDeviceSynchronize()); */

	/* Copy `theModel` to replicas */
	for (id = first; id < ctx->modelmanager->size; ++id) {

		if (ctx->modelmanager->locked[id] && ctx->modelmanager->replicas[id]->dev == defaultDev->id) {

			checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, ctx->modelmanager->replicas[id]->server, 0));

			dbg("Copy the model to model replica #%d\n", id);
			crossbowDataBufferCopyDeviceRegion
				(ctx->modelmanager->replicas[id]->data, defaultBaseModel->data, defaultDev->modelSynchronisationStream);

			/* Record event. This ought to remove the need to synchronise on `ctx->modelSynchronisationStream` */
			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
		}
	}

	/* checkCudaErrors(cudaDeviceSynchronize()); */

	return;
}

static void crossbowSynchronisationMultiGPUDefault (crossbowExecutionContextP ctx, int first, int clock) {

	(void)   ctx;
	(void) first;
	(void) clock;
	
	err ("Multi-GPU default SGD model synchronisation is not supported yet");
	return;
}

void crossbowSynchronisationDefault (crossbowExecutionContextP ctx, int first, int clock) {
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUDefault (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUDefault (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}