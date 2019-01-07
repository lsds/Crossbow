#include "kernelscalar.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "databuffer.h"

crossbowKernelScalarP crossbowKernelScalarCreate () {
	crossbowKernelScalarP p;
	p = (crossbowKernelScalarP) crossbowMalloc (sizeof(crossbow_kernel_scalar_t));
	p->name = NULL;
	p->value = NULL;
	p->type = UND;
	return p;
}

void crossbowKernelScalarSetIntValue (crossbowKernelScalarP p, const char *str, int i) {
	p->name = crossbowStringCopy (str);
	p->value = crossbowDataBufferCreate (sizeof(int), PIN); /* Create both host and device pointers */
	*(int *)(crossbowDataBufferGetHostPointer(p->value)) = i;
	p->type = I32;
	crossbowDataBufferPushSync(p->value);
	return;
}

int crossbowKernelScalarGetIntValue (crossbowKernelScalarP p) {
	nullPointerException (p->value);
	return *(int *)(crossbowDataBufferGetHostPointer(p->value));
}

int *crossbowKernelScalarGetDeviceBufferAsInt (crossbowKernelScalarP p) {
	nullPointerException (p->value);
	return (int *)(crossbowDataBufferGetDevicePointer(p->value));
}

void  crossbowKernelScalarSetFloatValue (crossbowKernelScalarP p, const char *str, float f) {
	p->name = crossbowStringCopy (str);
	p->value = crossbowDataBufferCreate (sizeof(float), PIN); /* Create both host and device pointers */
	*(float *)(crossbowDataBufferGetHostPointer(p->value)) = f;
	p->type = F32;
	crossbowDataBufferPushSync(p->value);
	return;
}

float crossbowKernelScalarGetFloatValue (crossbowKernelScalarP p) {
	nullPointerException (p->value);
	return *(float *)(crossbowDataBufferGetHostPointer(p->value));
}

float *crossbowKernelScalarGetDeviceBufferAsFloat (crossbowKernelScalarP p) {
	nullPointerException (p->value);
	return (float *)(crossbowDataBufferGetDevicePointer(p->value));
}

void  crossbowKernelScalarSetDoubleValue (crossbowKernelScalarP p, const char *str, double d) {
	p->name = crossbowStringCopy (str);
	p->value = crossbowDataBufferCreate (sizeof(double), PIN); /* Create both host and device pointers */
	*(double *)(crossbowDataBufferGetHostPointer(p->value)) = d;
	p->type = F64;
	crossbowDataBufferPushSync(p->value);
	return;
}

double crossbowKernelScalarGetDoubleValue (crossbowKernelScalarP p) {
	nullPointerException (p->value);
	return *(double *)(crossbowDataBufferGetHostPointer(p->value));
}

double *crossbowKernelScalarGetDeviceBufferAsDouble (crossbowKernelScalarP p) {
	nullPointerException (p->value);
	return (double *)(crossbowDataBufferGetDevicePointer(p->value));
}

static const char *typeToString (crossbowKernelScalar_t type) {
	     if (type == I32) return    "int";
	else if (type == F32) return  "float";
	else if (type == F64) return "double";
	else
		return NULL;
}

char *crossbowKernelScalarString (crossbowKernelScalarP p) {
	char s [256];
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	crossbowStringAppend (s, &offset, &remaining, "%s %s = ", typeToString (p->type), p->name);

	switch (p->type) {

	case I32: crossbowStringAppend (s, &offset, &remaining,    "%d", crossbowKernelScalarGetIntValue   (p)); break;
	case F32: crossbowStringAppend (s, &offset, &remaining, "%5.2f", crossbowKernelScalarGetFloatValue (p)); break;
	case F64: crossbowStringAppend (s, &offset, &remaining, "%5.2f", crossbowKernelScalarGetDoubleValue(p)); break;
	case UND:
		illegalStateException ();
	}
	return crossbowStringCopy (s);
}

void crossbowKernelScalarFree (crossbowKernelScalarP p) {
	if (! p)
		return;
	crossbowStringFree (p->name);
	crossbowDataBufferFree (p->value);
	crossbowFree (p, sizeof(crossbow_kernel_scalar_t));
	return;
}

