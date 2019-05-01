#include "gradientdescentoptimiser.h"

#include "optimisers/default.h"
#include "optimisers/downpour.h"
#include "optimisers/easgd.h"
#include "optimisers/hogwild.h"
#include "optimisers/polyakruppert.h"
#include "optimisers/sma.h"
#include "optimisers/synchronouseasgd.h"
#include "optimisers/synchronoussgd.h"

void crossbowKernelGradientDescentOptimiser (void *args) {

#ifdef UPDATE_MODEL_INCREMENTALLY
	/* Do nothing */
 	(void) args;
#else
	crossbowStreamP s = (crossbowStreamP) args;

	switch (s->model->type) {

	case DEFAULT:
		dbg("Update replica using default model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case  MULTI_GPU:
			crossbowKernelOptimiserDefault (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case WORKER:
		dbg("Update replica using S-SGD model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case  MULTI_GPU:
			crossbowKernelOptimiserSynchronousSGD (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case EAMSGD:
		dbg("Update replica using asynchronous EAMSGD model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelOptimiserElasticAveragingSGD (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case SYNCHRONOUSEAMSGD:
		dbg("Update replica using synchronous EAMSGD model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelOptimiserSynchronousElasticAveragingSGD (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	case DOWNPOUR:
		dbg("Update replica using (synchronous) DOWNPOUR model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelOptimiserDownpour (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;
	
	case HOGWILD:
		dbg("Update replica using Hogwild! model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelOptimiserHogwild (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;
	
	case POLYAK_RUPPERT:
		dbg("Update replica using Polyak-Ruppert model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelOptimiserPolyakRuppert (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;
		
	case SMA:
		dbg("Update replica using SMA model\n");
		switch (s->mode) {
		case SINGLE_GPU:
		case MULTI_GPU:
			crossbowKernelOptimiserSMA  (s);
			break;
		default:
			err("Invalid model synchronisation mode\n");
		}
		break;

	default:
		err("Invalid model update type\n");
	}

#endif /* UPDATE_MODEL_INCREMENTALLY */

	return;
}
