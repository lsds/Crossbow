#include "localvariable.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "device.h"

crossbowLocalVariableP crossbowLocalVariableCreate (const char *name, crossbowVariableP variable, int readonly, int replicationfactor, crossbowArrayListP devices) {
	int deviceId;
	int numberofdevices;
	crossbowDeviceP dev;

	crossbowVariableP var;

	crossbowThetaQueueP queue;

	crossbowLocalVariableP p;
	dbg("Register %s local variable %s (replication factor is %d)\n", (readonly) ? "read-only" : "read-write", name, replicationfactor);
	p = (crossbowLocalVariableP) crossbowMalloc (sizeof(crossbow_localvariable_t));

	p->name = crossbowStringCopy (name);
	p->type = (! readonly) ? RW : RO;

	p->theVariable = variable;

	/* Get number of devices */
	numberofdevices = crossbowArrayListSize (devices);

	p->variables = crossbowArrayListCreate (numberofdevices);

	for (deviceId = 0; deviceId < numberofdevices; deviceId++) {
		dev = crossbowArrayListGet (devices, deviceId);
		if (! crossbowDeviceSelected(dev))
			continue;
		crossbowArrayListSet (p->variables, dev->id, crossbowVariableReplicate (p->theVariable, NULL));
	}

	/*
	 * If a local variable is read-only, we initialise its GPU buffer once.
	 * The GPU buffer is accessible by all scheduled GPU execution streams.
	 *
	 * Otherwise, we allocate a GPU-only buffer for each active stream.
	 */
	if (p->type == RO) {

		for (deviceId = 0; deviceId < numberofdevices; deviceId++) {
			dev = crossbowArrayListGet (devices, deviceId);
			if (! crossbowDeviceSelected(dev))
				continue;
			
			/* Making sure buffer creation is redirected to the appropriate device */
			checkCudaErrors (cudaSetDevice(dev->id));
			
			var = crossbowArrayListGet (p->variables, dev->id);
			crossbowVariableSetDataBuffer (var, crossbowDataBufferCreate (var->schema->bytes, PIN), 0);
		}
		p->pool = NULL;
	} else {

		p->pool = crossbowArrayListCreate (numberofdevices);

		for (deviceId = 0; deviceId < numberofdevices; deviceId++) {
			dev = crossbowArrayListGet (devices, deviceId);
			if (! crossbowDeviceSelected(dev))
				continue;
			var = crossbowArrayListGet (p->variables, dev->id);
			queue = crossbowThetaQueueCreate (replicationfactor);
			#ifndef __LAZY_MATERIALISATION
			/* Making sure buffer creation is redirected to the appropriate device */
			checkCudaErrors (cudaSetDevice (dev->id));
			int ndx;
			for (ndx = 0; ndx < replicationfactor; ++ndx)
				crossbowThetaQueueSet (queue, ndx, (void *) crossbowDataBufferCreate (var->schema->bytes, REF));
			#endif
			crossbowArrayListSet (p->pool, dev->id, queue);
		}
	}
	return p;
}

crossbowDataBufferP crossbowLocalVariableGetDataBuffer (crossbowLocalVariableP p, int dev, int index, int *offset, int *length) {

	crossbowDataBufferP buffer = NULL;

	crossbowVariableP var = (crossbowVariableP) crossbowArrayListGet (p->variables, dev);

	if (p->type == RW) {

		crossbowThetaQueueP queue = (crossbowThetaQueueP) crossbowArrayListGet (p->pool, dev);

		dbg("Get local variable data buffer (device %d, index %d)\n", dev, index);

		buffer = crossbowThetaQueueGet (queue, index);
		/* Lazy materialisation */
		if (! buffer) {
			dbg("Create new data buffer for kernel local variable %s (%d bytes) on device %d\n", p->name, var->schema->bytes, dev);
			buffer = crossbowDataBufferCreate (var->schema->bytes, REF);
			/* The slot in the queue is already reserved, so setting at this point is safe */
			crossbowThetaQueueSet (queue, index, buffer);
		}
		buffer->queue = queue;
		buffer->index = index;
		invalidConditionException(buffer->refs == 0);
		/* Set reference counter to 1 */
		buffer->refs ++;
		/* Store offset and length */
		if (offset)
			*offset = 0;
		if (length)
			*length = var->schema->bytes;
	} else {
		/* Read-only local variables are already materialised and initialised */
		buffer = crossbowVariableGetDataBuffer (var, offset, length);
		buffer->queue = NULL;
		buffer->index = 0;
	}

	return buffer;
}

void crossbowLocalVariableResizePool (crossbowLocalVariableP p, crossbowArrayListP devices) {
	int deviceId;
	int numberofdevices;
	crossbowDeviceP dev;

	crossbowThetaQueueP queue;

	nullPointerException (p);
	nullPointerException (p->pool);

	/* Get number of devices */
	numberofdevices = crossbowArrayListSize (devices);

	for (deviceId = 0; deviceId < numberofdevices; deviceId++) {
		dev = crossbowArrayListGet (devices, deviceId);
		if (! crossbowDeviceSelected(dev))
			continue;

		queue = (crossbowThetaQueueP) crossbowArrayListGet (p->pool, dev->id);

		#ifndef __LAZY_MATERIALISATION
		illegalStateException ();
		#endif
		crossbowThetaQueueExpand (queue, NULL);
	}
}


int crossbowLocalVariableReadOnly (crossbowLocalVariableP p) {
	return (p->type == RO);
}

char *crossbowLocalVariableString (crossbowLocalVariableP p) {
	nullPointerException(p);
	char s [1024];
	char *t;
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	t = crossbowVariableString (p->theVariable);
	crossbowStringAppend (s, &offset, &remaining, "\"%s\" %s (%s)", p->name, t, (p->type == RO) ? "r/o" : "r/w");
	crossbowStringFree (t);
	return crossbowStringCopy (s);
}

void crossbowLocalVariableFree (crossbowLocalVariableP p) {
	if (! p)
		return;
	int i, j;
	int size;
	crossbowThetaQueueP queue;
	crossbowVariableP var;
	crossbowDataBufferP buffer;
	size = crossbowArrayListSize (p->variables);
	if (p->type == RW) {
		if (p->pool) {
			invalidConditionException (size == crossbowArrayListSize (p->pool));
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
			crossbowArrayListFree (p->pool);
		}
	} else {
		for (i = 0; i < size; ++i) {
			var = (crossbowVariableP) crossbowArrayListGet (p->variables, i);
			if (var) {
				buffer = crossbowVariableGetDataBuffer (var, NULL, NULL);
				crossbowDataBufferFree (buffer);
			}
		}
	}
	for (i = 0; i < size; ++i) {
		var = (crossbowVariableP) crossbowArrayListGet (p->variables, i);
		if (var)
			crossbowVariableFree (var);
	}
	crossbowArrayListFree (p->variables);
	crossbowVariableFree (p->theVariable);
	crossbowStringFree (p->name);
	crossbowFree (p, sizeof(crossbow_localvariable_t));
	return;
}
