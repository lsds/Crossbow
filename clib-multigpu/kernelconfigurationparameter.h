#ifndef __CROSSBOW_KERNEL_CONFIGURATION_PARAMETER_H_
#define __CROSSBOW_KERNEL_CONFIGURATION_PARAMETER_H_

#include "utils.h"

typedef struct crossbow_kernel_configuration_parameter *crossbowKernelConfigParamP;
typedef struct crossbow_kernel_configuration_parameter {
	char *name;
	void *value;
	int bytes;
	crossbowKernelConfigParam_t type;
} crossbow_kernel_configuration_parameter_t;

crossbowKernelConfigParamP crossbowKernelConfigParamCreate ();

void crossbowKernelConfigParamSetIntValue (crossbowKernelConfigParamP, const char *, int);
int  crossbowKernelConfigParamGetIntValue (crossbowKernelConfigParamP);

void  crossbowKernelConfigParamSetFloatValue (crossbowKernelConfigParamP, const char *, float);
float crossbowKernelConfigParamGetFloatValue (crossbowKernelConfigParamP);

void crossbowKernelConfigParamSetIntArray (crossbowKernelConfigParamP, const char *, int *, int);
int *crossbowKernelConfigParamGetIntArray (crossbowKernelConfigParamP, int *);

void   crossbowKernelConfigParamSetFloatArray (crossbowKernelConfigParamP, const char *, float *, int);
float *crossbowKernelConfigParamGetFloatArray (crossbowKernelConfigParamP, int *);

void   crossbowKernelConfigParamSetDoubleValue (crossbowKernelConfigParamP, const char *, double);
double crossbowKernelConfigParamGetDoubleValue (crossbowKernelConfigParamP);

char *crossbowKernelConfigParamString (crossbowKernelConfigParamP);

void crossbowKernelConfigParamFree (crossbowKernelConfigParamP);

#endif /* __CROSSBOW_KERNEL_CONFIGURATION_PARAMETER_H_ */
