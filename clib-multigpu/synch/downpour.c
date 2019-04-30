#include "downpour.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUDownpour (crossbowExecutionContextP ctx, int first, int clock) {
	(void)   ctx;
	(void) first;
	(void) clock;
	err ("Single-GPU DOWNPOUR SGD model synchronisation is not supported yet");
	return;
}

static void crossbowSynchronisationMultiGPUDownpour (crossbowExecutionContextP ctx, int first, int clock) {
	(void)   ctx;
	(void) first;
	(void) clock;
	err ("Multi-GPU DOWNPOUR SGD model synchronisation is not supported yet");
	return;
}

void crossbowSynchronisationDownpour (crossbowExecutionContextP ctx, int first, int clock) {
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUDownpour (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUDownpour  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
