#include "stream.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h" /* Defines use of pinned memory */

#include <limits.h>

crossbowStreamP crossbowStreamCreate (int id, crossbowDeviceP dev, int ops, int splits, crossbowVariableSchemaP examples, crossbowVariableSchemaP labels, int branches, crossbowModelSynchronisationMode_t mode, unsigned long long seed) {
	int i;
	int bytes;
	crossbowStreamP p = (crossbowStreamP) crossbowMalloc (sizeof(crossbow_stream_t));

	p->id = id;
	p->deviceId = dev->id;

	p->mode = mode;

	/* All calls directed to a particular device */
	checkCudaErrors(cudaSetDevice(p->deviceId));
	
	p->branches = branches;
	invalidConditionException(p->branches > 0);

	p->stream = crossbowMalloc (p->branches * sizeof(cudaStream_t));
	for (i = 0; i < p->branches; ++i)
		checkCudaErrors(cudaStreamCreateWithFlags(&(p->stream[i]), cudaStreamNonBlocking));

#ifdef INTRA_TASK_MEASUREMENTS
	/* Enable timing measurements and create start event */
	/*
	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&p->start, cudaEventBlockingSync));
	*/
	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventDefault));
	checkCudaErrors(cudaEventCreateWithFlags(&p->start, cudaEventDefault));
#else
	/*
	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventBlockingSync | cudaEventDisableTiming));
	*/
	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventDefault | cudaEventDisableTiming));
#endif

#ifdef MAKESPAN_MEASUREMENTS
	p->barrier = dev->barrier;
#endif

	p->cublasHandle = crossbowMalloc (p->branches * sizeof(cublasHandle_t));
	p->cudnnHandle = crossbowMalloc (p->branches * sizeof(cudnnHandle_t));
	/* There is one cuRAND generator per stream */
	checkCurandStatus(curandCreateGenerator(&(p->curandGenerator), CURAND_RNG_PSEUDO_DEFAULT));
	/* Set seed */
	checkCurandStatus(curandSetPseudoRandomGeneratorSeed(p->curandGenerator, seed));
	for (i = 0; i < p->branches; ++i) {
		/* cuBLAS handler */
		checkCublasStatus(cublasCreate(&(p->cublasHandle[i])));
		/* cuDNN handler */
		checkCudnnStatus(cudnnCreate(&(p->cudnnHandle[i])));
		/* Set cuBLAS stream. All subsequent cuBLAS function calls ought to be scheduled on stream `p->stream[i]` */
		checkCublasStatus(cublasSetStream (p->cublasHandle[i], p->stream[i]));
		/* Set cuDNN stream. All subsequent cuDNN function call ought to be scheduled on stream `p->stream[i]` */
		checkCudnnStatus(cudnnSetStream (p->cudnnHandle[i], p->stream[i]));
	}

	p->splits = splits;

	/*
	 * The shape of variable examples already takes into account
	 * the fact that we may split the batch into smaller tasks.
	 *
	 * The size, however reflect the entire batch plus padding.
	 */
	bytes = examples->bytes + labels->bytes;


#ifdef __INPUT_ISPINNED_
	p->input = crossbowDataBufferCreate (bytes, PIN);
#else
	p->input = crossbowDataBufferCreate (bytes, REF);
#endif

	p->examples = crossbowVariableCreate (crossbowVariableSchemaCopy (examples));
	p->labels   = crossbowVariableCreate (crossbowVariableSchemaCopy (labels));

	/* Set input data buffer pointers */
	crossbowVariableSetDataBuffer (p->examples, p->input, 0);
	crossbowVariableSetDataBuffer (p->labels,   p->input, examples->bytes);

	p->ops = ops;
	p->outputs = (crossbowListP *) crossbowMalloc (p->ops * sizeof (crossbowListP));
	for (i = 0; i < p->ops; ++i)
		p->outputs[i] = crossbowListCreate ();

	p->locals = (crossbowListP *) crossbowMalloc (p->ops * sizeof (crossbowListP));
	for (i = 0; i < p->ops; ++i)
		p->locals[i] = crossbowListCreate();

	p->theModel = NULL;

	p->modelSynchronisationHandle = NULL; /* dev->modelSynchronisationHandle; */
	p->modelSynchronisationStream = NULL; /* dev->modelSynchronisationStream; */

	crossbowStreamClear (p);

	return p;
}

crossbowDataBufferP crossbowStreamOperatorGetInput (crossbowStreamP p, crossbowOperatorP op) {
	int order;
	crossbowOperatorP prev;
	crossbowDataBufferP buffer;
	/*
	 * We cannot use `crossbowDataflowMostUpstream(p->dataflow, op)` here because
	 * it skips the data transformation operator and returns the raw input buffer.
	 */
	if (op->upstream == NULL) {
		return crossbowVariableGetDataBuffer (p->examples, NULL, NULL);
	}
	else {
		/* Input is the output of the previous operator.
		 * Check that there is only one, and get its id.
		 */
		invalidConditionException (crossbowArrayListSize(op->upstream) == 1);
		prev = crossbowArrayListGet (op->upstream, 0);
		/* Check that previous operator has produced at
		 * least one output. */
		invalidConditionException (crossbowListSize(p->outputs [prev->id]) > 0);

		if (crossbowListSize(p->outputs [prev->id]) == 1) {
			dbg("Return as input the output of %d\n", prev->id);
			return crossbowListPeekHead(p->outputs [prev->id]);
		}
		else {
			/* Previous operator has produced more than one outputs.
			 * Try to find current operator in its downstream nodes.
			 */
			for (order = 0; order < crossbowArrayListSize(prev->downstream); ++order) {
				if (op == crossbowArrayListGet (prev->downstream, order)) {
					/* Operator found. Lookup corresponding output */
					buffer = (crossbowDataBufferP) crossbowListPeek (p->outputs [prev->id], order);
					invalidConditionException (buffer != NULL);
					return buffer;
				}
			}
			/* If not found, throw exception */
			illegalStateException ();
		}
	}
}

/**
 * Get input of the current operator
 */
crossbowDataBufferP crossbowStreamGetCurrentInput (crossbowStreamP p) {
	return crossbowStreamOperatorGetInput (p, p->op);
}

/**
 * Get input of the current operator's peer
 */
crossbowDataBufferP crossbowStreamGetPeerInput (crossbowStreamP p) {
	nullPointerException(p->op->peer);
	return crossbowStreamOperatorGetInput (p, p->op->peer);
}

crossbowDataBufferP crossbowStreamOperatorGetOutput (crossbowStreamP p, crossbowOperatorP op) {
	int order;
	crossbowDataBufferP buffer;
	if (crossbowListSize(p->outputs [op->id]) == 1) {
		return crossbowListPeekHead(p->outputs [op->id]);
	}
	else {
		/* Operator `op` has produced more than one outputs. Try to find
		 * current operator (`p->op`) in `op`'s list of downstream nodes.
		 */
		for (order = 0; order < crossbowArrayListSize(op->downstream); ++order) {
			if (p->op == crossbowArrayListGet (op->downstream, order)) {
				/* Operator found. Lookup corresponding output */
				buffer = (crossbowDataBufferP) crossbowListPeek (p->outputs [op->id], order);
				invalidConditionException (buffer != NULL);
				return buffer;
			}
		}
		/* If not found, throw exception */
		illegalStateException ();
	}
}

crossbowDataBufferP crossbowStreamGetCurrentOutput (crossbowStreamP p) {
	crossbowDataBufferP output = NULL;
	if (! crossbowOperatorGetOutputBufferFromElsewhere(p->op)) {
		output = crossbowKernelGetOutputBuffer (p->op->kernel, p->deviceId, p->id);
	}
	else {
		dbg("Return the output of node %d (%s), position is %d\n", p->op->provider->id, p->op->provider->kernel->name, p->op->position);
		output = (crossbowDataBufferP) crossbowListPeek (p->outputs [p->op->provider->id], p->op->position);
		/* Increment reference counter */
		output->refs++;
	}
	return output;
}

/**
 * Get output of the current operator's peer
 */
crossbowDataBufferP crossbowStreamGetPeerOutput (crossbowStreamP p) {
	nullPointerException(p->op->peer);
	/* Assert that peer has produced at most one output */
	invalidConditionException (crossbowListSize(p->outputs[p->op->peer->id]) >= 1);
	return crossbowListPeekHead(p->outputs [p->op->peer->id]);
}

void crossbowStreamComputeInputCheckSum (crossbowStreamP p) {
	int i, j;
	int s; /* Step size */
	int b, n;
	float image;
	int label;
	float imagetotal = 0;
	int labeltotal = 0;
	if (! crossbowDataflowMostUpstream (p->dataflow, p->op))
		err("Fatal error\n");
	b = p->examples->schema->shape[0];
	n = p->examples->schema->bytes;
	if ((n % b) != 0)
		err("Fatal error\n");
	s = n / b;
	/* Assert that the labels' buffer is the same as the one for examples */
	if (p->examples->buffer != p->labels->buffer)
		err("Fatal error\n");
	if (p->examples->buffer != p->input)
		err("Fatal error\n");
	j = 0;
	for (i = 0; i < n; i += s) {
		image = crossbowDataBufferComputeCheckSum (p->input, i, s);
		label = crossbowDataBufferComputeCheckSumAsInt (p->input, (p->labels->offset + (j * 4)), 4);
		/* info("%2d: %+7d %+15.5f\n", j++, label, image); */
		imagetotal += image;
		labeltotal += label;
	}
	info("Total: %+7d %+15.5f\n", labeltotal, imagetotal);
}

void crossbowStreamComputeCheckSum (crossbowStreamP p) {

	float checksum;
	crossbowDataBufferP output;

	/* Compute input checksum */
	if (crossbowDataflowMostUpstream (p->dataflow, p->op))
	{
		checksum = crossbowDataBufferComputeCheckSum (p->input, 0, p->examples->schema->bytes);
		info("Kernel's %s input checksum is %.5f\n", p->op->kernel->name, checksum);
	}

	/* Compute output checksum */
	if (crossbowListEmpty(p->outputs[p->op->id])) {

		info("Kernel's %s output is null\n", p->op->kernel->name);
	}
	else {
		output = crossbowListPeekHead (p->outputs[p->op->id]);
		nullPointerException(output);
		checksum = crossbowDataBufferComputeCheckSum (output, 0, p->op->kernel->output->schema->bytes);
		info("Kernel's %s output (%p) checksum is %.5f\n", p->op->kernel->name, output, checksum);
	}
	return;
}

#ifdef SHARD_AXPY
static void crossbowStreamShardedSaxpy
	(cublasHandle_t handle, int elements, float *alpha, float *x, int incX, float *y, int incY) {

	/*
	 * Assume elements are floats (4 bytes):
	 *   0.125 MB contain    32768 elements
	 *   0.25  MB contain    65536 elements
	 *   0.5   MB contain   131072 elements
	 *   1.0   MB contains  262144 elements
	 *   2.0   MB contain   524288 elements
	 *   4.0   MB contain  1048576 elements
	 *   8.0   MB contain  2097152 elements
	 *  16.0   MB contain  4194304 elements
	 *  32.0   MB contain  8388608 elements
	 */
	int partition = 83886608;
	int remaining;
	float *px;
	float *py;
	int offset;

	if (elements <= partition) {
		checkCublasStatus(cublasSaxpy (handle, elements, alpha, x, incX, y, incY));
	}
	else {
		dbg("Sharding saxpy into %d calls\n", (((elements - (elements % partition)) / partition) + 1));
		remaining = elements;
		px = x;
		py = y;
		while (remaining > 0) {
			offset = (remaining > partition) ? partition : remaining;
			checkCublasStatus(cublasSaxpy (handle, offset, alpha, px, incX, py, incY));
			/* Increment pointers */
			px = px + offset;
			py = py + offset;
			/* Decrement remaining */
			remaining -= offset;
		}
	}
}
#endif

static void crossbowStreamSaxpy (cublasHandle_t handle, int elements, float *alpha, float *x, int incX, float *y, int incY) {
#ifdef SHARD_AXPY
	crossbowStreamShardedSaxpy (handle, elements, alpha, x, incX, y, incY);
#else
	checkCublasStatus(cublasSaxpy (handle, elements, alpha, x, incX, y, incY));
#endif
}

#ifdef UPDATE_MODEL_INCREMENTALLY
static void crossbowStreamDefaultModelUpdate (crossbowStreamP p, int count) {
	int order;
	int offset, length;
	int elements;
	float rate;

	crossbowDataBufferP buffer;

	float *data;
	float *gradient;
	float *last;
	float *base;

	float minusone = -1;

	/* Synchronise one or more model replica variables */
	for (order = 1; order <= count; ++order) {

		buffer = crossbowModelVariable (p->model, p->op->peer->kernel->id, order, &offset, &length);

		/* Assumes elements are floats */
		elements = length / 4;

		data     = (float *) ((char *) (buffer->dev)             + offset); /* Local replica model pointers */
		gradient = (float *) ((char *) (p->model->gradient->dev) + offset);
		last     = (float *) ((char *) (p->model->last->dev)     + offset);

		base     = (float *) ((char *) (p->theModel->data->dev)  + offset); /* Base model pointers */

		if (p->model->weightDecay > 0) {
			/* Add biased model variable to gradient */
			dbg("Compute gradient update with weight decay\n");
			crossbowStreamSaxpy (p->handle, elements, &(p->model->weightDecay), data, 1, gradient, 1);
		}

		if (p->model->momentum > 0) {

			rate = crossbowModelGetLearningRateForVariable(p->model, p->task, p->op->peer->id, order);
			
			/* Scale gradient based on learning rate */
			checkCublasStatus(cublasSscal(p->handle, elements, &(rate), gradient, 1));

			/* Apply momentum to gradient */
			crossbowStreamSaxpy (p->handle, elements, &(p->model->momentum), last, 1, gradient, 1);

			/* Copy current gradient into last */
			checkCudaErrors(cudaMemcpyAsync(last, gradient, length, cudaMemcpyDeviceToDevice, p->stream));

			/* Record event that gradient is ready to be used by parameter server */
			checkCudaErrors(cudaEventRecord (p->model->client[p->op->peer->id], p->stream));

			/* Apply gradient to local model */
			crossbowStreamSaxpy (p->handle, elements, &(minusone), gradient, 1, data, 1);

			/* Apply gradient to parameter server model (base model) */
			checkCudaErrors(cudaStreamWaitEvent(p->modelSynchronisationStream, p->model->client[p->op->peer->id], 0));
			crossbowStreamSaxpy (p->modelSynchronisationHandle, elements, &(minusone), gradient, 1, base, 1);
			checkCudaErrors(cudaEventRecord(p->model->server[p->op->peer->id], p->modelSynchronisationStream));

		} else {

			rate = - crossbowModelGetLearningRateForVariable(p->model, p->task, p->op->peer->id, order);
			
			/*
			info("offset   at %d\n", offset);
			info("length   is %d (or %d elements)\n", length, elements);
			info("data     at %p\n", data);
			info("gradient at %p\n", gradient);
			*/

			/* Record event that gradient is ready to be used by parameter server */
			checkCudaErrors(cudaEventRecord (p->model->client[p->op->peer->id], p->stream));

			/* Apply gradient to local model */
			crossbowStreamSaxpy (p->handle, elements, &(rate), gradient, 1, data, 1);

			/* Apply gradient to parameter server model (base model) */
			checkCudaErrors(cudaStreamWaitEvent(p->modelSynchronisationStream, p->model->client[p->op->peer->id], 0));
			crossbowStreamSaxpy (p->modelSynchronisationHandle, elements, &(rate), gradient, 1, base, 1);
			checkCudaErrors(cudaEventRecord(p->model->server[p->op->peer->id], p->modelSynchronisationStream));
		}
	}

	return;
}

static void crossbowStreamWorkerModelUpdate (crossbowStreamP p, int count) {

	(void) count;

	/* Record event that gradient is ready to be used by parameter server */
	checkCudaErrors(cudaEventRecord (p->model->client[p->op->peer->id], p->stream));

	return;
}

static void crossbowStreamSynchronousEamsgdModelUpdate (crossbowStreamP p, int count) {
	int order;
	int offset, length;
	int elements;
	float rate;

	crossbowDataBufferP buffer;

	float *data;
	float *gradient;
	float *last;

	float minusone = -1;

	/* Synchronise one or more model replica variables */
	for (order = 1; order <= count; ++order) {

		buffer = crossbowModelVariable (p->model, p->op->peer->kernel->id, order, &offset, &length);

		/* Assumes elements are floats */
		elements = length / 4;

		data     = (float *) ((char *) (buffer->dev)             + offset); /* Local replica model pointers */
		gradient = (float *) ((char *) (p->model->gradient->dev) + offset);
		last     = (float *) ((char *) (p->model->last->dev)     + offset);

		if (p->model->weightDecay > 0) {
			/* Add biased model variable to gradient */
			crossbowStreamSaxpy (p->handle, elements, &(p->model->weightDecay), data, 1, gradient, 1);
		}

		if (p->model->momentum > 0) {

			rate = crossbowModelGetLearningRateForVariable(p->model, p->task, p->op->peer->id, order);

			/* Scale gradient based on learning rate */
			checkCublasStatus(cublasSscal(p->handle, elements, &(rate), gradient, 1));

			/* Apply momentum to gradient */
			crossbowStreamSaxpy (p->handle, elements, &(p->model->momentum), last, 1, gradient, 1);

			/* Copy current gradient into last */
			checkCudaErrors(cudaMemcpyAsync(last, gradient, length, cudaMemcpyDeviceToDevice, p->stream));

			/* Apply gradient to local model */
			crossbowStreamSaxpy (p->handle, elements, &(minusone), gradient, 1, data, 1);

		} else {

			rate = - crossbowModelGetLearningRateForVariable(p->model, p->task, p->op->peer->id, order);

			/* Apply gradient to local model */
			crossbowStreamSaxpy (p->handle, elements, &(rate), gradient, 1, data, 1);
		}
	}

	return;
}

static void crossbowStreamEamsgdModelUpdate (crossbowStreamP p, int count) {
    (void) p;
    (void) count;
    err ("Incremental asynchronous elastic averaging SGD is not supported yet");
}

#endif /* UPDATE_MODEL_INCREMENTALLY */

void crossbowStreamUpdateModel (crossbowStreamP p) {

#ifndef UPDATE_MODEL_INCREMENTALLY
	/* Do nothing */
	(void) p;
#else
	int count;
	if (p->op->peer == NULL)
		return;
	/* Are there any model variables updated? */
	count = crossbowModelVariableCount (p->model, p->op->peer->kernel->id);
	if (count == 0)
		return;

	switch (s->model->type) {

	case DEFAULT:
		dbg("Update replica incrementally using default model\n");
		crossbowStreamDefaultModelUpdate (p, count);
		break;

	case WORKER:
		dbg("Update replica incrementally using worker model\n");
		crossbowStreamWorkerModelUpdate  (p, count);
		break;

	case EAMSGD:
		dbg("Update replica incrementally using EAMSGD model\n");
		crossbowStreamEamsgdModelUpdate  (p, count);
		break;

	case SYNCHRONOUSEAMSGD:
		dbg("Update replica incrementally using synchronous EAMSGD model\n");
		crossbowStreamSynchronousEamsgdModelUpdate  (p, count);
		break;

	default:
		err("Invalid model update type\n");
	}
#endif
	return;
}

void crossbowStreamClear (crossbowStreamP p) {
	int i;

	for (i = 0; i < p->ops; i++) {

		if (! crossbowListEmpty(p->outputs[i]))
			illegalStateException();

		if (! crossbowListEmpty(p->locals [i]))
			illegalStateException();
	}

	p->task = 0;
	p->phi = TRAIN;

	p->freeP[0] = LONG_MIN;
	p->freeP[1] = LONG_MIN;

	p->model = NULL;
	p->dataflow = NULL;
	p->op = NULL;

	return;
}

void crossbowStreamFree (crossbowStreamP p) {
	int i;
	for (i = 0; i < p->branches; ++i) {
		checkCudaErrors(cudaStreamDestroy(p->stream[i]));
		/* Free stream handlers */
		checkCublasStatus(cublasDestroy(p->cublasHandle[i]));
		checkCudnnStatus(cudnnDestroy(p->cudnnHandle[i]));
	}
	crossbowFree (p->stream, (p->branches * sizeof(cudaStream_t)));
	crossbowFree (p->cublasHandle, (p->branches * sizeof(cublasHandle_t)));
	crossbowFree (p->cudnnHandle, (p->branches * sizeof(cudnnHandle_t)));
	checkCurandStatus(curandDestroyGenerator(p->curandGenerator));
	checkCudaErrors(cudaEventDestroy(p->event));
	crossbowVariableFree (p->examples);
	crossbowVariableFree (p->labels);
	crossbowDataBufferFree (p->input);
	for (i = 0; i < p->ops; i++)
		crossbowListFree (p->outputs[i]);
	crossbowFree (p->outputs, p->ops * sizeof (crossbowDataBufferP));
	for (i = 0; i < p->ops; i++)
		crossbowListFree (p->locals[i]);
	crossbowFree (p->locals, p->ops * sizeof (crossbowDataBufferP));
	crossbowFree (p, sizeof(crossbow_stream_t));
}
