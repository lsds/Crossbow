#ifndef __CROSSBOW_KERNELMAP_H_
#define __CROSSBOW_KERNELMAP_H_

#include "utils.h"

typedef struct crossbow_kernel_binding *crossbowKernelBindingP;
typedef struct crossbow_kernel_binding {
	crossbowKernelBindingP next;
	char *name;
	crossbowKernelFunctionP func;
} crossbow_kernel_binding_t;

typedef struct crossbow_kernel_map *crossbowKernelMapP;
typedef struct crossbow_kernel_map {
	crossbowKernelBindingP *bin;
	int slots;
	int count;
} crossbow_kernel_map_t;

crossbowKernelMapP crossbowKernelMapCreate (int);

void crossbowKernelMapBind (crossbowKernelMapP, char *name, crossbowKernelFunctionP);

crossbowKernelFunctionP crossbowKernelMapResolve (crossbowKernelMapP, const char *name);

void crossbowKernelMapDump (crossbowKernelMapP);

void crossbowKernelMapFree (crossbowKernelMapP);

#endif /* __CROSSBOW_KERNELMAP_H_ */
