#include "hogwild.h"

#include "common.h"

static void crossbowSynchronisationSingleGPUHogwild (crossbowExecutionContextP ctx, int first, int clock) {
	(void)   ctx;
	(void) first;
	(void) clock;
	err ("Single-GPU Hogwild! SGD model synchronisation is not supported yet");
	return;
}

static void crossbowSynchronisationMultiGPUHogwild (crossbowExecutionContextP ctx, int first, int clock) {
	(void)   ctx;
	(void) first;
	(void) clock;
	err ("Multi-GPU Hogwild! SGD model synchronisation is not supported yet");
	return;
}

void crossbowSynchronisationHogwild (crossbowExecutionContextP ctx, int first, int clock) {
	switch (ctx->mode) {
		case SINGLE_GPU:
			crossbowSynchronisationSingleGPUHogwild (ctx, first, clock);
			break;
		case MULTI_GPU:
			crossbowSynchronisationMultiGPUHogwild  (ctx, first, clock);
			break;
		default:
			err("Invalid model synchronisation mode\n");
	}
	return;
}
