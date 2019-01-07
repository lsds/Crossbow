#include "kernelconfigurationparameter.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowKernelConfigParamP crossbowKernelConfigParamCreate () {
	crossbowKernelConfigParamP p;
	p = (crossbowKernelConfigParamP) crossbowMalloc (sizeof(crossbow_kernel_configuration_parameter_t));
	p->name = NULL;
	p->value = NULL;
	p->bytes = 0;
	p->type = UNDEFINED;
	return p;
}

void crossbowKernelConfigParamSetIntValue (crossbowKernelConfigParamP p, const char *str, int i) {
	p->name = crossbowStringCopy (str);
	p->bytes = sizeof(int);
	p->value = (int *) crossbowMalloc (p->bytes);
	*(int *)(p->value) = i;
	p->type = INT;
	return;
}

int crossbowKernelConfigParamGetIntValue (crossbowKernelConfigParamP p) {
	nullPointerException (p->value);
	return *(int *)(p->value);
}

void  crossbowKernelConfigParamSetFloatValue (crossbowKernelConfigParamP p, const char *str, float f) {
	p->name = crossbowStringCopy (str);
	p->bytes = sizeof(float);
	p->value = (float *) crossbowMalloc (p->bytes);
	*(float *)(p->value) = f;
	p->type = FLOAT;
	return;
}

float crossbowKernelConfigParamGetFloatValue (crossbowKernelConfigParamP p) {
	nullPointerException (p->value);
	return *(float *)(p->value);
}

void crossbowKernelConfigParamSetIntArray (crossbowKernelConfigParamP p, const char *str, int *arr, int len) {
	int i;
	p->name = crossbowStringCopy (str);
	p->bytes = len * sizeof(int);
	p->value = crossbowMalloc (p->bytes);
	for (i = 0; i < len; ++i)
		((int *) p->value)[i] = arr[i];
	p->type = INT_ARRAY;
	return;
}

int *crossbowKernelConfigParamGetIntArray (crossbowKernelConfigParamP p, int *len) {
	if (len)
		*len = p->bytes / sizeof(int);
	return (int *) (p->value);
}

void crossbowKernelConfigParamSetFloatArray (crossbowKernelConfigParamP p, const char *str, float *arr, int len) {
	int i;
	p->name = crossbowStringCopy (str);
	p->bytes = len * sizeof(float);
	p->value = crossbowMalloc (p->bytes);
	for (i = 0; i < len; ++i)
		((float *) p->value)[i] = arr[i];
	p->type = FLOAT_ARRAY;
	return;
}

float *crossbowKernelConfigParamGetFloatArray (crossbowKernelConfigParamP p, int *len) {
	if (len)
		*len = p->bytes / sizeof(float);
	return (float *) (p->value);
}

void  crossbowKernelConfigParamSetDoubleValue (crossbowKernelConfigParamP p, const char *str, double d) {
	p->name = crossbowStringCopy (str);
	p->bytes = sizeof(double);
	p->value = (double *) crossbowMalloc (p->bytes);
	*(double *)(p->value) = d;
	p->type = DOUBLE;
	return;
}

double crossbowKernelConfigParamGetDoubleValue (crossbowKernelConfigParamP p) {
	nullPointerException (p->value);
	return *(double *)(p->value);
}

static const char *typeToString (crossbowKernelConfigParam_t type) {
	if (type == INT) return "int";
	else
	if (type == FLOAT) return "float";
	else
	if (type == INT_ARRAY) return "int []";
	else
	if (type == FLOAT_ARRAY) return "float []";
	else
	if (type == DOUBLE) return "double";
	else
		return NULL;
}

char *crossbowKernelConfigParamString (crossbowKernelConfigParamP p) {
	int i;
	int length;
	int *ints;
	float *floats;

	char s [256];
	int offset;
	size_t remaining;

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	crossbowStringAppend (s, &offset, &remaining, "%s %s = ", typeToString (p->type), p->name);

	switch (p->type) {

	case INT:    crossbowStringAppend (s, &offset, &remaining,    "%d", crossbowKernelConfigParamGetIntValue    (p)); break;
	case FLOAT:  crossbowStringAppend (s, &offset, &remaining, "%5.2f", crossbowKernelConfigParamGetFloatValue  (p)); break;
	case DOUBLE: crossbowStringAppend (s, &offset, &remaining, "%5.4f", crossbowKernelConfigParamGetDoubleValue (p)); break;

	case INT_ARRAY:

		length = 0;
		ints = crossbowKernelConfigParamGetIntArray(p, &length);

		crossbowStringAppend (s, &offset, &remaining, "[");
		for (i = 0; i < length; i++) {
			crossbowStringAppend (s, &offset, &remaining, "%d", ints[i]);
			if (i < (length - 1))
				crossbowStringAppend (s, &offset, &remaining, ", ");
			else
				crossbowStringAppend (s, &offset, &remaining, "]");
		}

		break;

	case FLOAT_ARRAY:

		length = 0;
		floats = crossbowKernelConfigParamGetFloatArray(p, &length);

		crossbowStringAppend (s, &offset, &remaining, "[");
		for (i = 0; i < length; i++) {
			crossbowStringAppend (s, &offset, &remaining, "%.3f", floats[i]);
			if (i < (length - 1))
				crossbowStringAppend (s, &offset, &remaining, ", ");
			else
				crossbowStringAppend (s, &offset, &remaining, "]");
		}

		break;
	case UNDEFINED:
		illegalStateException ();
	}
	return crossbowStringCopy (s);
}

void crossbowKernelConfigParamFree (crossbowKernelConfigParamP p) {
	if (! p)
		return;
	crossbowStringFree (p->name);
	crossbowFree (p->value, p->bytes);
	crossbowFree (p, sizeof(crossbow_kernel_configuration_parameter_t));
	return;
}

