#include "cudnntensor.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

crossbowCudnnTensorP crossbowCudnnTensorCreate (int count, int channels, int height, int width) {

	crossbowCudnnTensorP p;

	p = (crossbowCudnnTensorP) crossbowMalloc (sizeof(crossbow_cudnn_tensor_t));
	checkCudnnStatus(cudnnCreateTensorDescriptor(&(p->descriptor)));

	/* Stride between two consecutive columns */
	int wStride = 1;
	/* Stride between two consecutive rows */
	int hStride = wStride * width;
	/* Stride between two consecutive channels */
	int cStride = hStride * height;
	/* Stride between two consecutive images */
	int nStride = cStride * channels;

	checkCudnnStatus(cudnnSetTensor4dDescriptorEx(p->descriptor, CUDNN_DATA_FLOAT, count, channels, height, width, nStride, cStride, hStride, wStride));

	return p;
}

char *crossbowCudnnTensorString (crossbowCudnnTensorP p) {

	cudnnDataType_t type;
	int n, c, h, w;
	int nStride, cStride, hStride, wStride;

	char s [1024];
	int offset;
	size_t remaining;

	checkCudnnStatus(cudnnGetTensor4dDescriptor (p->descriptor, &type, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	/* [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "[%d, %d, %d, %d]", n, c, h, w);

	return crossbowStringCopy (s);
}

void crossbowCudnnTensorFree (crossbowCudnnTensorP p) {

	if (! p)
		return;

	checkCudnnStatus(cudnnDestroyTensorDescriptor(p->descriptor));
	crossbowFree (p, sizeof(crossbow_cudnn_tensor_t));

	return;
}
