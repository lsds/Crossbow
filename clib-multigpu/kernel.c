#include "kernel.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "localvariable.h"
#include "kernelconfigurationparameter.h"
#include "kernelscalar.h"

#include "device.h"

#include <stdio.h>

crossbowKernelP crossbowKernelCreate (int id, const char *name, crossbowKernelFunctionP func, int numberofinputs, int numberofvariables, int numberofoutputs, int pull) {
	int i;
	crossbowKernelP p;
	p = (crossbowKernelP) crossbowMalloc (sizeof(crossbow_kernel_t));
	p->id = id;
	p->name = crossbowStringCopy (name);
	p->func = func;

	p->number_of_inputs = numberofinputs;
	p->inputs = (crossbowVariableP *) crossbowMalloc (p->number_of_inputs * sizeof(crossbowVariableP));
	for (i = 0; i < p->number_of_inputs; i++)
		p->inputs[i] = NULL;

	if (numberofvariables > 0)
		p->variables = crossbowArrayListCreate (numberofvariables);
	else
		p->variables = NULL;

	p->parameters = NULL; /* Initialised by TheGPU_setKernelConfigurationParameters() in GPU.c */

	p->scalars = NULL; /* Initialised by TheGPU_setKernelScalars() in GPU.c */

	p->number_of_outputs = numberofoutputs;
	p->output = NULL;
	p->pool = NULL;

	/* This helps identify those output buffers that contain loss and accuracy values.
	 * If true, the kernel output buffers are pinned, so that the output can be pulled
	 * from GPU memory. */
	p->pull = pull;

	p->cudnnKernelType = NONE;

	return p;
}

int crossbowKernelOutputPull (crossbowKernelP p) {
	return (p->pull > 0);
}

void crossbowKernelSetInput (crossbowKernelP p, int ndx, crossbowVariableSchemaP schema) {
	indexOutOfBoundsException (ndx, p->number_of_inputs);
	if (p->inputs[ndx]) {
		fprintf(stderr, "error: kernel %s's input %d already set\n", p->name, ndx);
		exit(1);
	}
	p->inputs[ndx] = crossbowVariableCreate (schema);
	return;
}

void crossbowKernelSetOutput (crossbowKernelP p, crossbowVariableSchemaP schema) {
	if (p->output) {
		fprintf(stderr, "error: kernel %s's output already set\n", p->name);
		exit(1);
	}
	p->output = crossbowVariableCreate (schema);
	return;
}

void crossbowKernelSetOutputBufferPool (crossbowKernelP p, int replicationfactor, crossbowArrayListP devices) {
	int deviceId;
	int numberofdevices;
	crossbowDeviceP dev;

	crossbowThetaQueueP queue;

	numberofdevices = crossbowArrayListSize (devices);

	invalidConditionException(p->number_of_outputs == 1);

	p->pool = crossbowArrayListCreate (numberofdevices);

	for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {

		dev = crossbowArrayListGet (devices, deviceId);
		if (! crossbowDeviceSelected(dev))
			continue;

		queue = crossbowThetaQueueCreate (replicationfactor * (p->number_of_outputs));

		#ifndef __LAZY_MATERIALISATION
		/* Making sure buffer creation is redirected to the appropriate device */
		checkCudaErrors (cudaSetDevice (dev->id));
		int idx;
		for (idx = 0; idx < replicationfactor; ++idx)
			crossbowThetaQueueSet (queue, idx, (crossbowDataBufferCreate (p->output->schema->bytes, (p->pull ? PIN : REF))));
		#endif
		crossbowArrayListSet (p->pool, dev->id, queue);
	}
	return;
}

crossbowDataBufferP crossbowKernelGetOutputBuffer (crossbowKernelP p, int dev, int index) {
	crossbowDataBufferP buffer;

	crossbowThetaQueueP queue = (crossbowThetaQueueP) crossbowArrayListGet (p->pool, dev);

	dbg("Get output data buffer (device %d, index %d)\n", dev, index);

	buffer = crossbowThetaQueueGet (queue, index);
	/*
	 * Lazy materialisation of output buffers: note that `cudaMalloc` and `cudaFree`
	 * are implicit GPU synchronisation points.
	 *
	 * Also note that when this function is called the appropriate device is already
	 * set.
	 */
	if (! buffer) {
		dbg("Create new output buffer for kernel %s (size is %d pull is %d) on device %d\n", p->name, p->output->schema->bytes, p->pull, dev);
		buffer = crossbowDataBufferCreate (p->output->schema->bytes, (p->pull ? PIN : REF));
		/* The slot in the queue is already reserved, so setting at this point is safe */
		crossbowThetaQueueSet (queue, index, buffer);
	}
	buffer->queue = queue;
	buffer->index = index;
	invalidConditionException (buffer->refs == 0);
	/* Increment reference counter */
	buffer->refs ++;
	return buffer;
}

void crossbowKernelResizeOutputBufferPool (crossbowKernelP p, crossbowArrayListP devices) {
	int deviceId;
	int numberofdevices;
	crossbowDeviceP dev;

	crossbowThetaQueueP queue;

	numberofdevices = crossbowArrayListSize (devices);

	for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {
		dev = crossbowArrayListGet (devices, deviceId);
		if (! crossbowDeviceSelected(dev))
			continue;

		queue = (crossbowThetaQueueP) crossbowArrayListGet (p->pool, dev->id);

		#ifndef __LAZY_MATERIALISATION
		illegalStateException();
		#endif
		crossbowThetaQueueExpand (queue, NULL);
	}
}

char *crossbowKernelString (crossbowKernelP p) {
	/* kernel `name` {`x` inputs: [] (), [] (), ... } {output: [] () } {`y` local variables: [] (), ... } {`z` parameters: ...} */
	int i;
	char s [1024];
	char *t;
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	crossbowStringAppend (s, &offset, &remaining, "kernel %s {%d input%s: ", p->name, p->number_of_inputs, (p->number_of_inputs == 1) ? "" : "s");
	for (i = 0; i < p->number_of_inputs; i++) {
		t = crossbowVariableString (p->inputs[i]);
		crossbowStringAppend (s, &offset, &remaining, "%s", t);
		crossbowStringFree (t);
		if (i < (p->number_of_inputs - 1))
			crossbowStringAppend (s, &offset, &remaining, ", ");
	}
	if (p->output) {
		t = crossbowVariableString (p->output);
		crossbowStringAppend (s, &offset, &remaining, "} {output: %s} ", t);
		crossbowStringFree (t);
	} else {
		crossbowStringAppend (s, &offset, &remaining, "} {0 outputs} ");
	}
	/* Append local variables */
	if (p->variables) {
		int size = crossbowArrayListSize (p->variables);
		info("%d local variables in %s\n", size, p->name);
		crossbowStringAppend (s, &offset, &remaining, "{%d local variable%s: ", size, (size == 1) ? "" : "s");
		for (i = 0; i < size; ++i) {
			crossbowLocalVariableP var = (crossbowLocalVariableP) crossbowArrayListGet (p->variables, i);
			if (var == NULL) {
				crossbowStringAppend (s, &offset, &remaining, "%s", "null");
			} else {
				t = crossbowLocalVariableString (var);
				crossbowStringAppend (s, &offset, &remaining, "%s", t);
				crossbowStringFree (t);
			}
			if (i < (size - 1))
				crossbowStringAppend (s, &offset, &remaining, ", ");
		}
		crossbowStringAppend (s, &offset, &remaining, "} ");
	}
	/* Append configuration parameters */
	if (p->parameters) {
		int size = crossbowArrayListSize (p->parameters);
		crossbowStringAppend (s, &offset, &remaining, "{%d parameter%s: ", size, (size == 1) ? "" : "s");
		for (i = 0; i < size; ++i) {
			crossbowKernelConfigParamP param = (crossbowKernelConfigParamP) crossbowArrayListGet (p->parameters, i);
			t = crossbowKernelConfigParamString (param);
			crossbowStringAppend (s, &offset, &remaining, "%s", t);
			crossbowStringFree (t);
			if (i < (size - 1))
				crossbowStringAppend (s, &offset, &remaining, ", ");
		}
		crossbowStringAppend (s, &offset, &remaining, "} ");
	}
	return crossbowStringCopy (s);
}

void crossbowKernelFree (crossbowKernelP p) {
	int i, j;
	int size;
	crossbowThetaQueueP queue;
	crossbowDataBufferP buffer;
	crossbowLocalVariableP variable;
	crossbowKernelConfigParamP parameter;
	crossbowKernelScalarP scalar;

	if (! p)
		return;

	crossbowStringFree (p->name);

	for (i = 0; i < p->number_of_inputs; i++)
		crossbowVariableFree (p->inputs[i]);
	crossbowFree (p->inputs, p->number_of_inputs * sizeof(crossbowVariableP));

	if (p->pool) {
		size = crossbowArrayListSize (p->pool);
		for (i = 0; i < size; ++i) {
			queue = (crossbowThetaQueueP) crossbowArrayListGet (p->pool, i);
			if (queue) {
				for (j = 0; j < crossbowThetaQueueSize (queue); ++j) {
					buffer = (crossbowDataBufferP) crossbowThetaQueueGet (queue, j);
					if (buffer)
						crossbowDataBufferFree (buffer);
				}
				crossbowThetaQueueFree (queue);
			}
		}
	}
	crossbowArrayListFree (p->pool);

	crossbowVariableFree (p->output);

	/* Free local variables */
	if (p->variables) {
		size = crossbowArrayListSize (p->variables);
		for (i = 0; i < size; ++i) {
			variable = (crossbowLocalVariableP) crossbowArrayListGet (p->variables, i);
			crossbowLocalVariableFree (variable);
		}
		crossbowArrayListFree (p->variables);
	}

	/* Free configuration parameters */
	if (p->parameters) {
		size = crossbowArrayListSize (p->parameters);
		for (i = 0; i < size; ++i) {
			parameter = (crossbowKernelConfigParamP) crossbowArrayListGet (p->parameters, i);
			crossbowKernelConfigParamFree (parameter);
		}
		crossbowArrayListFree (p->parameters);
	}

	/* Free kernel scalars */
	if (p->scalars) {
		size = crossbowArrayListSize (p->scalars);
		for (i = 0; i < size; ++i) {
			scalar = (crossbowKernelScalarP) crossbowArrayListGet (p->scalars, i);
			crossbowKernelScalarFree (scalar);
		}
		crossbowArrayListFree (p->scalars);
	}

	/* Free cuDNN descriptors */
	switch (p->cudnnKernelType) {
		case CONV:
			crossbowCudnnConvParamsFree (p->descriptors.conv);
			break;
		case POOL:
			crossbowCudnnPoolParamsFree (p->descriptors.pool);
			break;
		case RELU:
			crossbowCudnnReLUParamsFree (p->descriptors.relu);
			break;
		case SOFTMAX:
			crossbowCudnnSoftMaxParamsFree (p->descriptors.softmax);
			break;
		case BATCHNORM:
			crossbowCudnnBatchNormParamsFree (p->descriptors.batchnorm);
			break;
		case DROPOUT:
			crossbowCudnnDropoutParamsFree (p->descriptors.dropout);
		default:
			break;
	}

	crossbowFree (p, sizeof(crossbow_kernel_t));
	return;
}
