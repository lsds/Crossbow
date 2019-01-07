#include "model.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowModelP crossbowModelCreate (int ops, int bytes, int dev, int base) {
	dbg("Create model with %d bytes on device %d\n", bytes, dev);
	int i;
	crossbowModelP p;
	p = (crossbowModelP) crossbowMalloc (sizeof(crossbow_model_t));
	p->ops = ops;
	p->dev = dev;
	p->variables = (crossbowVariableP *) crossbowMalloc (p->ops * sizeof(crossbowVariableP));
	for (i = 0; i < p->ops; ++i)
		p->variables[i] = NULL;
	p->vars = 0;
	p->updates = 0;
	p->clock = 0;
	p->offset = 0;
	p->elements = 0;
	p->bytes = bytes;
    
    p->base = base;

	/* Redirect all CUDA calls to the correct device */
	checkCudaErrors (cudaSetDevice(p->dev));

	p->data = crossbowDataBufferCreate(p->bytes, PIN); /* Not a phantom variable */

	/* The model's gradient is allocated only on GPU memory. */
	p->gradient = crossbowDataBufferCreate(p->bytes, REF);
	checkCudaErrors(cudaMemset (p->gradient->dev, 0, p->bytes));

	p->last = NULL;
	p->diff = NULL;
    
    p->temp = NULL;

#ifdef ADAGRAD
	p->hist = crossbowDataBufferCreate(p->bytes, REF); 
	p->tmp1 = crossbowDataBufferCreate(p->bytes, REF);
    /* Reset buffers */
    checkCudaErrors(cudaMemset (p->hist->dev, 0, p->bytes));
	checkCudaErrors(cudaMemset (p->tmp1->dev, 0, p->bytes));
#endif

	/* Model configuration */
	p->wpc = 0;
	p->type = DEFAULT;
	p->conf = crossbowSolverConfCreate ();

	/* The model lock */
	pthread_mutex_init (&(p->lock), NULL);

	/* GPU events to synchronise with the parameter server */

#ifdef UPDATE_MODEL_INCREMENTALLY

	p->client = crossbowMalloc (p->ops * sizeof(cudaEvent_t));
	p->server = crossbowMalloc (p->ops * sizeof(cudaEvent_t));

	for (i = 0; i < p->ops; ++i) {
        /*
		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
        */
		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventDisableTiming));
	}
#else
    /*
	checkCudaErrors(cudaEventCreateWithFlags(&(p->client),  cudaEventBlockingSync | cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&(p->server),  cudaEventBlockingSync | cudaEventDisableTiming));
    */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->client), cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&(p->server), cudaEventDisableTiming));
#endif

	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventBlockingSync | cudaEventDisableTiming)); */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventDisableTiming));

	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventBlockingSync | cudaEventDisableTiming)); */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventDisableTiming));

	return p;
}

void crossbowModelFinalise (crossbowModelP model) {
    
    /*
     * We only finalise the first model ever configured. That model
     * also serves as a base model of the default device.
     */
    invalidConditionException(model->base > 0);
    
	/* Redirect all CUDA calls to the correct device */
	checkCudaErrors (cudaSetDevice(model->dev));

	/* The model's `diff` buffer is allocated only for base models and only if we use model averaging */
	/*
	 * Modified on 3 Aug. 2018: all replicas have a diff.
	 * 
	if (model->type == SYNCHRONOUSEAMSGD && model->base) {
        
		model->diff = crossbowDataBufferCreate (model->bytes, REF);
		checkCudaErrors(cudaMemset (model->diff->dev, 0, model->bytes));
	}
	*/
	model->diff = crossbowDataBufferCreate (model->bytes, REF);
	checkCudaErrors(cudaMemset (model->diff->dev, 0, model->bytes));
	
    
	/* The model's last gradient is allocated only if momentum is greater than 0. */
	if (model->conf->momentum > 0) {
        
		model->last = crossbowDataBufferCreate(model->bytes, REF);
		checkCudaErrors(cudaMemset (model->last->dev, 0, model->bytes));
	}
    
    /* Only the master model has a `temp` buffer allocated */
    model->temp = crossbowDataBufferCreate(model->bytes, REF);
    checkCudaErrors(cudaMemset (model->temp->dev, 0, model->bytes));
}

void crossbowModelRegister (crossbowModelP model, int ndx, int order, crossbowVariableSchemaP schema) {
	crossbowVariableP v, p;
	int ord;
	v = crossbowVariableCreate (schema);
	/* Set pointer to data buffer */
	crossbowVariableSetDataBuffer (v, model->data, model->offset);
	ord = 1;
	p = model->variables[ndx];
	if (p == NULL) {
		model->variables[ndx] = v;
	} else {
		ord = 2;
		while (p->next != NULL) {
			p = p->next;
			ord ++;
		}
		p->next = v;
		v->next = NULL;
	}
	/* Check order */
	if (ord != order) {
		fprintf (stderr, "error: invalid model variable order (ndx=%d, ord=%d)\n", ndx, ord);
		exit (1);
	}
	model->vars ++;
	/* Increment global pointer */
	model->offset += schema->bytes;
	/* Increment total number of elements */
	model->elements += schema->elements;
	return;
}

float crossbowModelGetLearningRateForVariable (crossbowModelP p, int task, int ndx, int order) {
	float rate;
	crossbowVariableP v;

	nullPointerException(p);

	rate = crossbowSolverConfGetLearningRate (p->conf, task);

	if (crossbowSolverConfHasIrregularLearningRate(p->conf)) {

		v = crossbowModelFind (p, ndx, order);

		/* Scale rate by multiplier */
		dbg("Model variable multiplier is %.5f\n", v->learningRateMultiplier);
		rate *= v->learningRateMultiplier;
	}

	return rate;
}

int crossbowModelClock (crossbowModelP p) {
	return p->clock;
}

void crossbowModelLock (crossbowModelP p) {
	pthread_mutex_lock(&(p->lock));
}

int crossbowModelTryLock (crossbowModelP p) {
	/* Returns 0 is successful, EBUSY if the lock is already acquired by another thread */
	return pthread_mutex_trylock(&(p->lock));
}

void crossbowModelUnlock (crossbowModelP p) {
	pthread_mutex_unlock(&(p->lock));
}

void crossbowModelIncrementClock (crossbowModelP p, int clock) {
	p->clock = clock;
	p->updates = 0;
	return;
}

crossbowModelP crossbowModelReplicate (crossbowModelP model, int dev, int base) {
	int i;
	crossbowModelP p;
	crossbowVariableP v, w;

	p = (crossbowModelP) crossbowMalloc (sizeof(crossbow_model_t));

	p->ops = model->ops;
	p->dev = dev;
	p->variables = (crossbowVariableP *) crossbowMalloc (p->ops * sizeof(crossbowVariableP));

	p->vars = model->vars;
	p->updates = model->updates;
	p->clock = model->clock;
	p->offset = model->offset;
	p->elements = model->elements;
	p->bytes = model->bytes;
    
    p->base = base;

	/* Redirect all CUDA calls to the correct device */
	checkCudaErrors (cudaSetDevice(p->dev));

	p->data = crossbowDataBufferReplicate (model->data);
	p->gradient = crossbowDataBufferReplicate (model->gradient);

#ifdef ADAGRAD
	p->hist = crossbowDataBufferReplicate (model->hist);
	p->tmp1 = crossbowDataBufferReplicate (model->tmp1);
#endif

	if (model->last)
		p->last = crossbowDataBufferReplicate (model->last);
	else
		p->last = NULL;

	/*
	 * Modified on 3 Aug. 2018
	 * 
	if (model->diff && p->base)
		p->diff = crossbowDataBufferReplicate (model->diff);
	else
		p->diff = NULL;
	*/
	p->diff = crossbowDataBufferReplicate (model->diff);
	
    
    /* Only the first base model has a temporary buffer allocated */
    p->temp = NULL;
    
	p->wpc = model->wpc;
	p->type = model->type;
	p->conf = crossbowSolverConfReplicate (model->conf);

	/* Replicate model variables */
	for (i = 0; i < p->ops; ++i) {
		v = model->variables[i];
		if (! v)
			p->variables[i] = NULL;
		else {
			p->variables[i] = crossbowVariableReplicate (v, p->data);
			w = p->variables[i];
			while (v->next != NULL) {
				v = v->next;
				w->next = crossbowVariableReplicate (v, p->data);
				w = w->next;
			}
		}
	}

	/* Init model replica lock */
	pthread_mutex_init (&(p->lock), NULL);

	/* Init GPU events to synchronise with the parameter server */

#ifdef UPDATE_MODEL_INCREMENTALLY

	p->client = crossbowMalloc (p->ops * sizeof(cudaEvent_t));
	p->server = crossbowMalloc (p->ops * sizeof(cudaEvent_t));

	for (i = 0; i < p->ops; ++i) {
        /*
		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
        */
		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventDisableTiming));
	}
#else
    /*
    checkCudaErrors(cudaEventCreateWithFlags(&(p->client), cudaEventBlockingSync | cudaEventDisableTiming));
    checkCudaErrors(cudaEventCreateWithFlags(&(p->server), cudaEventBlockingSync | cudaEventDisableTiming));
    */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->client), cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&(p->server), cudaEventDisableTiming));
#endif

	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventBlockingSync | cudaEventDisableTiming)); */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventDisableTiming));

	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventBlockingSync | cudaEventDisableTiming)); */
	checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventDisableTiming));

	return p;
}

int crossbowModelVariableCount (crossbowModelP p, int ndx) {
	crossbowVariableP v;
	int count;
	indexOutOfBoundsException (ndx, p->ops);
	v = p->variables[ndx];
	if (v == NULL) {
		return 0;
	}
	count = 1;
	while (v->next != NULL) {
		v = v->next;
		count ++;
	}
	return count;
}

crossbowVariableP crossbowModelFind (crossbowModelP p, int ndx, int order) {
	crossbowVariableP v;
	int ord;
	dbg("Find model variable from model %p for op %d/%d order %d\n", p, ndx, p->ops, order);
	indexOutOfBoundsException (ndx, p->ops);
	v = p->variables[ndx];
	ord = 1;
	while (ord != order && v->next != NULL) {
		v = v->next;
		ord ++;
	}
	if (ord != order) {
		fprintf(stderr, "error: model variable not found (id %d, order %d)\n", ndx, order);
		exit (1);
	}
	return v;
}

crossbowDataBufferP crossbowModelVariable (crossbowModelP p, int ndx, int order, int *offset, int *length) {
	crossbowVariableP v;
	int ord;
	dbg("Get model variable from model %p for op %d/%d order %d\n", p, ndx, p->ops, order);
	indexOutOfBoundsException (ndx, p->ops);
	v = p->variables[ndx];
	ord = 1;
	while (ord != order && v->next != NULL) {
		v = v->next;
		ord ++;
	}
	if (ord != order) {
		fprintf(stderr, "error: model variable not found (id %d, order %d)\n", ndx, order);
		exit (1);
	}
	return crossbowVariableGetDataBuffer (v, offset, length);
}

/*
 * Note on 22.6.2017
 *
 * Nesterov-based gradient computation has been updated
 * in optimiser.
 *
 * No need to update model variables here.
 */
crossbowDataBufferP crossbowModelVariableAccelerated (crossbowModelP p, int ndx, int order, int *offset, int *length, cublasHandle_t handle) {
	(void) handle;
	crossbowDataBufferP data = crossbowModelVariable (p, ndx, order, offset, length);
	/*
	if (p->conf->momentum > 0 && p->conf->momentumMethod == NESTEROV) {
		float *last     = (float *) ((char *) (p->last->dev) + *offset);
		float *variable = (float *) ((char *) (p->data->dev) + *offset);
		int elements    = *length / sizeof(float);
		checkCublasStatus(cublasSaxpy (handle, elements, &(p->conf->momentum), last, 1, variable, 1));
	}
	*/
	return data;
}

crossbowDataBufferP crossbowModelVariableGradient (crossbowModelP p, int ndx, int order, int *offset, int *length) {
	crossbowDataBufferP data = crossbowModelVariable (p, ndx, order, offset, length);
	invalidArgumentException (data == p->data);
	/* Ignore model data; return model gradient instead */
	return p->gradient;
}

crossbowDataBufferP crossbowModelVariableLastGradient (crossbowModelP p, int ndx, int order, int *offset, int *length) {
	crossbowDataBufferP data = crossbowModelVariable (p, ndx, order, offset, length);
	invalidArgumentException (data == p->data);
	/* Ignore model data; return last model gradient instead */
	return p->last;
}

void crossbowModelStore (crossbowModelP p, const char *prefix) {
	char *f = crossbowStringConcat ("%s-data.dat", prefix);
	char *g = crossbowStringConcat ("%s-last.dat", prefix);
	crossbowDataBufferStore (p->data, f);
	if (p->last)
		crossbowDataBufferStore (p->last, g);
	crossbowStringFree (f);
	crossbowStringFree (g);
	return;
}

void crossbowModelLoad  (crossbowModelP p, const char *prefix) {
	char *f = crossbowStringConcat ("%s-data.dat", prefix);
	char *g = crossbowStringConcat ("%s-last.dat", prefix);
	crossbowDataBufferLoad (p->data, f);
	if (p->last)
		crossbowDataBufferLoad (p->last, g);
	crossbowStringFree (f);
	crossbowStringFree (g);
	return;
}

void crossbowModelFree (crossbowModelP p) {
	int i;
	crossbowVariableP v, w;
	for (i = 0; i < p->ops; ++i) {
		for (v = p->variables[i]; v != NULL; v = w) {
			w = v->next;
			crossbowVariableFree(v);
		}
	}
	crossbowDataBufferFree (p->data);
	crossbowDataBufferFree (p->gradient);

#ifdef ADAGRAD
	crossbowDataBufferFree (p->hist);
	crossbowDataBufferFree (p->tmp1);
#endif

	if (p->last)
		crossbowDataBufferFree (p->last);
	if (p->diff)
		crossbowDataBufferFree (p->diff);
	if (p->temp)
		crossbowDataBufferFree (p->temp);

	crossbowSolverConfFree (p->conf);

	crossbowFree(p->variables, p->ops * sizeof(crossbowVariableP));

#ifdef UPDATE_MODEL_INCREMENTALLY
	for (i = 0; i < p->ops; ++i) {
		checkCudaErrors(cudaEventDestroy(p->client[i]));
		checkCudaErrors(cudaEventDestroy(p->server[i]));
	}
	crossbowFree(p->client, p->ops * sizeof(cudaEvent_t));
	crossbowFree(p->server, p->ops * sizeof(cudaEvent_t));
#else
	checkCudaErrors(cudaEventDestroy(p->client));
	checkCudaErrors(cudaEventDestroy(p->server));
#endif

	checkCudaErrors(cudaEventDestroy(p->updated));
	checkCudaErrors(cudaEventDestroy(p->accumulated));

	crossbowFree(p, sizeof(crossbow_model_t));
	return;
}

void crossbowModelDump (crossbowModelP p) {
	int i;
	int ord;
	crossbowVariableP v;
	char *s;
	printf ("=== [Model: %d variables] ===\n", p->vars);
	for (i = 0; i < p->ops; ++i) {
		printf ("Op %2d: ", i);
		v = p->variables[i];
		if (! v) printf ("null\n");
		else {
			ord = 1;
			while (v != NULL) {
				s = crossbowVariableString (v);
				printf("undefined (order %d) (%s)", ord, s);
				crossbowStringFree (s);
				printf(" -> ");
				ord ++;
				v = v->next;
			}
			printf("null\n");
		}
	}
	printf ("=== [End of model dump] ===\n");
	fflush (stdout);
	return;
}
