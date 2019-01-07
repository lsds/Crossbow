#include "cudnndropoutparams.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "../databuffer.h"

#include "cudnnhelper.h"

crossbowCudnnDropoutParamsP crossbowCudnnDropoutParamsCreate (int replicas) {
	crossbowCudnnDropoutParamsP p;
	p = (crossbowCudnnDropoutParamsP) crossbowMalloc (sizeof(crossbow_cudnn_dropout_params_t));
	/* Initialise struct to 0 */
	memset (p, 0, sizeof(crossbow_cudnn_dropout_params_t));
	p->replicas = replicas;
	invalidConditionException (p->replicas > 0);
	p->dropoutDesc = (cudnnDropoutDescriptor_t *) crossbowMalloc (p->replicas * sizeof(cudnnDropoutDescriptor_t));
	/* Initialise array to 0 */
	memset (p->dropoutDesc, 0, (p->replicas * sizeof(cudnnDropoutDescriptor_t)));
	p->states = crossbowArrayListCreate (p->replicas);
	return p;
}

void crossbowCudnnDropoutParamsSetInputDescriptor (crossbowCudnnDropoutParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	p->input = crossbowCudnnTensorCreate (count, channels, height, width);
}

void crossbowCudnnDropoutParamsSetOutputDescriptor (crossbowCudnnDropoutParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	p->output = crossbowCudnnTensorCreate (count, channels, height, width);
}

void crossbowCudnnDropoutParamsSetDropoutDescriptor (crossbowCudnnDropoutParamsP p, int ndx, cudnnHandle_t handle, float dropout, unsigned long long seed) {
	size_t statesSize;
	nullPointerException(p);
	indexOutOfBoundsException (ndx, p->replicas);
	checkCudnnStatus(cudnnCreateDropoutDescriptor(&(p->dropoutDesc[ndx])));
	checkCudnnStatus(cudnnDropoutGetStatesSize(handle, &statesSize));
	invalidConditionException (statesSize > 0);
	/* Create new buffer */
	crossbowDataBufferP data = crossbowDataBufferCreate (statesSize, REF);
	checkCudnnStatus(cudnnSetDropoutDescriptor(p->dropoutDesc[ndx], handle, dropout, data->dev, statesSize, seed));
	/* Store buffer */
	crossbowArrayListSet (p->states, ndx, data);
	return;
}

size_t crossbowCudnnDropoutParamsGetReserveSpaceSize (crossbowCudnnDropoutParamsP p) {
	nullPointerException(p);
	checkCudnnStatus(cudnnDropoutGetReserveSpaceSize (p->input->descriptor, &(p->reserveSpaceSize)));
	return p->reserveSpaceSize;
}

char *crossbowCudnnDropoutParamsString (crossbowCudnnDropoutParamsP p) {
	char s [1024];
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	/* Get tensor descriptors */
	char *u = crossbowCudnnTensorString (p->input);
	char *v = crossbowCudnnTensorString (p->output);
	/* input [n, c, h, w] output [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "input %s output %s", u, v);
	crossbowStringFree (u);
	crossbowStringFree (v);
	return crossbowStringCopy (s);
}

void crossbowCudnnDropoutParamsFree (crossbowCudnnDropoutParamsP p) {
	int i;
	crossbowDataBufferP buffer;
	if (! p)
		return;
	crossbowCudnnTensorFree (p->input);
	crossbowCudnnTensorFree (p->output);
	/* Destroy descriptors */
	for (i = 0; i < p->replicas; ++i)
		checkCudnnStatus(cudnnDestroyDropoutDescriptor(p->dropoutDesc[i]));
	crossbowFree (p->dropoutDesc, (p->replicas * sizeof(cudnnDropoutDescriptor_t)));
	/* Free data buffers in p->states */
	invalidConditionException (p->replicas == crossbowArrayListSize (p->states));
	for (i = 0; i < p->replicas; ++i) {
		buffer = crossbowArrayListGet (p->states, i);
		crossbowDataBufferFree (buffer);
	}
	crossbowArrayListFree (p->states);
	crossbowFree (p, sizeof(crossbow_cudnn_dropout_params_t));
}
