#include "cudnnsoftmaxparams.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "cudnnhelper.h"

crossbowCudnnSoftMaxParamsP crossbowCudnnSoftMaxParamsCreate () {

	crossbowCudnnSoftMaxParamsP p;

	p = (crossbowCudnnSoftMaxParamsP) crossbowMalloc (sizeof(crossbow_cudnn_softmax_params_t));

	dbg("Softmax parameters @%p\n", p);
	/* Initialise to 0 */
	memset (p, 0, sizeof(crossbow_cudnn_softmax_params_t));

	return p;
}

void crossbowCudnnSoftMaxParamsSetInputDescriptor (crossbowCudnnSoftMaxParamsP p, int count, int channels, int height, int width) {

	p->input = crossbowCudnnTensorCreate (count, channels, height, width);
}

void crossbowCudnnSoftMaxParamsSetOutputDescriptor (crossbowCudnnSoftMaxParamsP p, int count, int channels, int height, int width) {

	p->output = crossbowCudnnTensorCreate (count, channels, height, width);
}

char *crossbowCudnnSoftMaxParamsString (crossbowCudnnSoftMaxParamsP p) {

	char s [1024];
	int offset;
	size_t remaining;

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	/* Get tensor descriptors */
	char *d1 = crossbowCudnnTensorString (p->input);
	char *d2 = crossbowCudnnTensorString (p->output);

	/* input [n, c, h, w] output [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "input %s output %s", d1, d2);

	crossbowStringFree (d1);
	crossbowStringFree (d2);

	return crossbowStringCopy (s);
}

void crossbowCudnnSoftMaxParamsFree (crossbowCudnnSoftMaxParamsP p) {

	if (! p)
		return;

	crossbowCudnnTensorFree (p->input);
	crossbowCudnnTensorFree (p->output);

	crossbowFree (p, sizeof(crossbow_cudnn_softmax_params_t));
}
