#ifndef __CROSSBOW_KERNEL_H_
#define __CROSSBOW_KERNEL_H_

#include "utils.h"

#include "variable.h"
#include "arraylist.h"
#include "thetaqueue.h"

#include "cudnn/cudnnconvparams.h"
#include "cudnn/cudnnpoolparams.h"
#include "cudnn/cudnnreluparams.h"
#include "cudnn/cudnnsoftmaxparams.h"
#include "cudnn/cudnnbatchnormparams.h"
#include "cudnn/cudnndropoutparams.h"

typedef struct crossbow_kernel *crossbowKernelP;
typedef struct crossbow_kernel {
	int id;
	char *name;
	crossbowKernelFunctionP func;

	int number_of_inputs;
	crossbowVariableP *inputs;

	/* Local variables */
	crossbowArrayListP variables;

	/* Scalars, mostly used by cuBLAS, like `alpha` and `beta` in `cublasSgemm()` */
	crossbowArrayListP scalars;

	/* Configuration parameters */
	crossbowArrayListP parameters;

	/* The number of downstream operators scales
	 * the output buffer queue size accordingly.
	 */
	int number_of_outputs;
	crossbowVariableP output;
	crossbowArrayListP pool; /* Array of queues, one per device */

	int pull;

	/* Is this kernel one of the cuDNN kernels (i.e. Convolution, Pooling, ReLU, or SoftMax)? */
	crossbowCudnnKernel_t cudnnKernelType;

	union {
		crossbowCudnnConvParamsP conv;
		crossbowCudnnPoolParamsP pool;
		crossbowCudnnReLUParamsP relu;
		crossbowCudnnSoftMaxParamsP softmax;
		crossbowCudnnBatchNormParamsP batchnorm;
		crossbowCudnnDropoutParamsP dropout;
	} descriptors;

} crossbow_kernel_t;

crossbowKernelP crossbowKernelCreate (int, const char *, crossbowKernelFunctionP, int, int, int, int);

void crossbowKernelSetInput (crossbowKernelP, int, crossbowVariableSchemaP);

void crossbowKernelSetOutput (crossbowKernelP, crossbowVariableSchemaP);

void crossbowKernelInitialiseConfigParams (crossbowKernelP, int);

void crossbowKernelSetOutputBufferPool (crossbowKernelP, int, crossbowArrayListP);

void crossbowKernelResizeOutputBufferPool (crossbowKernelP, crossbowArrayListP);

crossbowDataBufferP crossbowKernelGetOutputBuffer (crossbowKernelP, int, int);

int crossbowKernelOutputPull (crossbowKernelP);

char *crossbowKernelString (crossbowKernelP);

void crossbowKernelFree (crossbowKernelP);

#endif /* __CROSSBOW_KERNEL_H_ */
