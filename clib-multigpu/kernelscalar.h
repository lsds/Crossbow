#ifndef __CROSSBOW_KERNEL_SCALAR_H_
#define __CROSSBOW_KERNEL_SCALAR_H_

#include "utils.h"

#include "databuffer.h"

typedef struct crossbow_kernel_scalar *crossbowKernelScalarP;
typedef struct crossbow_kernel_scalar {
	char *name;
	crossbowDataBufferP value;
	crossbowKernelScalar_t type;
} crossbow_kernel_scalar_t;

crossbowKernelScalarP crossbowKernelScalarCreate ();

void crossbowKernelScalarSetIntValue (crossbowKernelScalarP, const char *, int);
int  crossbowKernelScalarGetIntValue (crossbowKernelScalarP);

void  crossbowKernelScalarSetFloatValue (crossbowKernelScalarP, const char *, float);
float crossbowKernelScalarGetFloatValue (crossbowKernelScalarP);

void   crossbowKernelScalarSetDoubleValue (crossbowKernelScalarP, const char *, double);
double crossbowKernelScalarGetDoubleValue (crossbowKernelScalarP);

int    *crossbowKernelScalarGetDeviceBufferAsInt    (crossbowKernelScalarP);
float  *crossbowKernelScalarGetDeviceBufferAsFloat  (crossbowKernelScalarP);
double *crossbowKernelScalarGetDeviceBufferAsDouble (crossbowKernelScalarP);

char *crossbowKernelScalarString (crossbowKernelScalarP);

void crossbowKernelScalarFree (crossbowKernelScalarP);

#endif /* __CROSSBOW_KERNEL_SCALAR_H_ */
