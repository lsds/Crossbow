#include "eamsgd.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
	(void)   ctx;
	(void) first;
	(void) clock;
	err ("Single-GPU asynchronous Elastic Averaging SGD model synchronisation is not supported yet\n");
	return;
}

static void crossbowSynchronisationMultiGPUElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
	(void)   ctx;
	(void) first;
	(void) clock;
	err ("Multi-GPU asynchronous Elastic Averaging SGD model synchronisation is not supported yet\n");
	return;
}

void crossbowSynchronisationElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUElasticAveragingSGD (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUElasticAveragingSGD  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
