#ifndef __CROSSBOW_CUDNN_TENSOR_H_
#define __CROSSBOW_CUDNN_TENSOR_H_

#include <cudnn.h>

typedef struct crossbowCudnnTensor *crossbowCudnnTensorP;
typedef struct crossbowCudnnTensor {

	cudnnTensorDescriptor_t descriptor;

} crossbow_cudnn_tensor_t;

crossbowCudnnTensorP crossbowCudnnTensorCreate (int, int, int, int);

char *crossbowCudnnTensorString (crossbowCudnnTensorP);

void crossbowCudnnTensorFree (crossbowCudnnTensorP);

#endif /* __CROSSBOW_CUDNN_TENSOR_H_ */
