#include "modelmanager.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "device.h"

#define SLOT_LENGTH 64
#define SLOT_OFFSET(x) (x * SLOT_LENGTH)

crossbowModelManagerP crossbowModelManagerCreate (JNIEnv *env, int size, crossbowModelP model, crossbowModelSynchronisation_t type, crossbowArrayListP devices) {
	int i, j;

	int deviceId;
	int numberofdevices;
	crossbowDeviceP dev;

	crossbowModelManagerP p = (crossbowModelManagerP) crossbowMalloc(sizeof(crossbow_modelmanager_t));

	p->theModel = model;

	p->size = 0;

	numberofdevices = crossbowArrayListSize (devices);

	p->baseModels = crossbowArrayListCreate (numberofdevices);
	for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {
		dev = crossbowArrayListGet (devices, deviceId);
		if (! crossbowDeviceSelected (dev))
			continue;
		if (dev->id == model->dev) {
			info("Device #%d uses the first model\n", dev->id);
			crossbowArrayListSet (p->baseModels, dev->id, model); /* Do not replicate the first model */
		} else {
			crossbowArrayListSet (p->baseModels, dev->id, crossbowModelReplicate (model, dev->id, 1));
		}
		/* Increase the number of available model replicas */
		p->size += size;
	}

	invalidArgumentException (p->size >= 1);

	p->replicas = (crossbowModelP *) crossbowMalloc (p->size * sizeof(crossbowModelP));
	/*
	 * It used to be that: `p->replicas[0] = model`. However,
	 * `model` now serves as the parameter server model.
	 */

	/* Create replicas in a round-robin fashion */
	i = 0;
	for (j = 0; j < size; ++j) {

		for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {

			dev = crossbowArrayListGet (devices, deviceId);
			if (! crossbowDeviceSelected (dev))
				continue;

			p->replicas[i++] = crossbowModelReplicate (model, dev->id, 0);
		}
	}
	invalidConditionException(i == p->size);

	/* Redirect calls to the appropriate device */
	checkCudaErrors (cudaSetDevice(p->theModel->dev));
	crossbowDataBufferPullSync (p->theModel->data);

	for (i = 0; i < p->size; ++i) {

		checkCudaErrors (cudaSetDevice(p->replicas[i]->dev));

		crossbowDataBufferPullSync (p->replicas[i]->data);

		if (memcmp(p->replicas[i]->data->host, p->theModel->data->host, p->replicas[i]->bytes) != 0) {
			fprintf(stderr, "error: buffers not equal in constructor\n");
			exit(1);
		}
	}
	/* Redirect calls to the default device */
	checkCudaErrors (cudaSetDevice(p->theModel->dev));

	/* Create multi-device synchronisation events on default device */
    p->synched = (cudaEvent_t *) crossbowMalloc (numberofdevices * sizeof(cudaEvent_t));
    for (i = 0; i < numberofdevices; ++i)
        checkCudaErrors(cudaEventCreateWithFlags(&(p->synched[i]), cudaEventDisableTiming));
        // checkCudaErrors(cudaEventCreateWithFlags(&(p->synched[i]), cudaEventBlockingSync | cudaEventDisableTiming));
    
    /*
     * Assign `synched` event to each device individually. This will help 
     * with overlapping inter-device operations but requires extra checks
     * to ensure correctness. 
     * 
     * (See use of NCCL library is executioncontext.c)
     *
	 * p->synched = (cudaEvent_t *) crossbowMalloc (numberofdevices * sizeof(cudaEvent_t));
	 * for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {
	 *	dev = crossbowArrayListGet (devices, deviceId);
	 *	if (! crossbowDeviceSelected (dev))
	 *		continue;
	 *	checkCudaErrors (cudaSetDevice(dev->id));
	 *	checkCudaErrors(cudaEventCreateWithFlags(&(p->synched[dev->id]), cudaEventBlockingSync | cudaEventDisableTiming));
	 * }
	 * checkCudaErrors (cudaSetDevice(p->theModel->dev));
     */

	/*
	 * Acquire global references to Java objects and their methods
	 */
	p->integerClassRef = (jclass) (*env)->NewGlobalRef (env, (*env)->FindClass (env, "java/lang/Integer"));

	p->constructor = (*env)->GetMethodID (env, p->integerClassRef,   "<init>", "(I)V");
	p->intValue    = (*env)->GetMethodID (env, p->integerClassRef, "intValue",  "()I");

	p->threadClassRef = (jclass) (*env)->NewGlobalRef (env, (*env)->FindClass (env, "java/lang/Thread"));
	p->yield = (*env)->GetStaticMethodID (env, p->threadClassRef, "yield", "()V");

	/* Initialise internal state */
	p->locked = (int *) crossbowMalloc (p->size * sizeof(int));

	p->queue = crossbowThetaQueueCreate (p->size);

	for (i = 0; i < p->size; ++i) {

		/* Reset model manager internal state */
		p->locked[i] = 0;

		/* Set model id */
		p->replicas[i]->id = i;

		/* Fill queue with pointers to model identifiers */
		crossbowThetaQueueSet (p->queue, i, &(p->replicas[i]->id));
	}

	/* Set synchronisation type: it is one of BSP, SSP, or ASP */
	p->type = type;
	
	p->disabled = 0;

	/* Initialise the model manager lock (used during auto-tuning) */
	pthread_mutex_init (&(p->lock), NULL);


	return p;
}

crossbowModelP crossbowModelManagerGetNextOrWait (JNIEnv *env, crossbowModelManagerP p, int bound) {
	int id;
	int *ptr = NULL;
	crossbowModelP replica;

	(void) env; /* Could be use to call Thread.yield() */

	ptr = (int *) crossbowThetaQueueGetNext (p->queue);
	id = *ptr;
	replica = p->replicas[id];
	/*
	 * Busy wait once more if the model's clock
	 * is not set correctly.
	 */
	while (bound > crossbowModelClock (replica)) {
        ;
	}
	/* Lock the model */
	crossbowModelLock(replica);
	return replica;
}

crossbowModelP crossbowModelManagerGet (JNIEnv *env, crossbowModelManagerP p, jobject replicaId) {
	int id;
	crossbowModelP replica;
	id = (*env)->CallIntMethod (env, replicaId, p->intValue);
	indexOutOfBoundsException (id, p->size);
	replica = p->replicas[id];
	/* Lock the model */
	crossbowModelLock(replica);
	return replica;
}

jobject crossbowModelManagerAcquireAccess (JNIEnv *env, crossbowModelManagerP p, int *clock) {
	jobject obj;
	int id;
	int *ptr = NULL;
	ptr = (int *) crossbowThetaQueueGetNext (p->queue);
	nullPointerException (ptr);
	id = *ptr;
	*clock = crossbowModelClock (p->replicas[id]);
	obj = (*env)->NewObject (env, p->integerClassRef, p->constructor, id);
	return obj;
}

jobject crossbowModelManagerUpgradeAccess (JNIEnv *env, crossbowModelManagerP p, jobject replicaId, int *clock) {
	int id;
	/* Update clock value for model `replicaId` */
	id = (*env)->CallIntMethod (env, replicaId, p->intValue);
	*clock = crossbowModelClock (p->replicas[id]);
	return replicaId;
}

void crossbowModelManagerRelease (crossbowModelManagerP p, crossbowModelP replica) {
	/* Unlock the model */
	crossbowModelUnlock (replica);
	crossbowThetaQueueRelease (p->queue, replica->id);
}

int crossbowModelManagerLockAll (crossbowModelManagerP p) {
	int count = crossbowModelManagerLockAny(p);
	return (count == p->size);
}

int crossbowModelManagerLockAny (crossbowModelManagerP p) {
	int i;
	int count = 0;
	for (i = 0; i < p->size; ++i) {
		p->locked[i] = 0; /* Reset counter */

		if (crossbowThetaQueueIsDisabled(p->queue, i)) {
			/* Trick system into thinking that BSP is preserved */
			// info("Replica #%d is disabled\n", i);
			++count;
			continue;
		}

		if (crossbowModelTryLock(p->replicas[i]) == 0) {
			p->locked[i] = 1;
			++count;
		}
	}
	dbg("%d/%d model replicas locked\n", count, p->size);
	return count;
}

int crossbowModelManagerUnlockAny (crossbowModelManagerP p) {
	int i;
	int count = 0;
	for (i = 0; i < p->size; ++i) {
		if (p->locked[i]) {
			crossbowModelUnlock(p->replicas[i]);
			p->locked[i] = 0;
			++count;
		}
	}
	dbg("%d/%d model replicas unlocked\n", count, p->size);
	return count;
}

int crossbowModelManagerNumberOfVisibleUpdates (crossbowModelManagerP p) {
	int i;
	int updates = 0;
	for (i = 0; i < p->size; ++i) {
		if (p->locked[i]) {
			dbg("Replica #%02d: %2d updates\n", i, p->replicas[i]->updates);
			updates += p->replicas[i]->updates;
		}
	}
	return updates;
}

void crossbowModelManagerIncrementClock (crossbowModelManagerP p, int clock) {
	int i;
	for (i = 0; i < p->size; ++i)
		if (p->locked[i])
			crossbowModelIncrementClock(p->replicas[i], clock);
	return;
}

int crossbowModelManagerLoad (crossbowModelManagerP p, const char *dir) {

	int id;
	crossbowModelP model;
	char *prefix;

	/* Load base models */
	for (id = 0; id < crossbowArrayListSize (p->baseModels); ++id) {

		model = (crossbowModelP) crossbowArrayListGet (p->baseModels, id);
		if (! model)
			continue;

		dbg("Override base model #%d on device %d\n", id, model->dev);

		/* Redirect CUDA calls to model device */
		checkCudaErrors (cudaSetDevice(model->dev));

		prefix = crossbowStringConcat ("%s/gpu-%02d-theModel", dir, model->dev);
		crossbowModelLoad (model, prefix);
		crossbowStringFree (prefix);
	}

	/* Load model replicas */
	for (id = 0; id < p->size; ++id) {

		dbg("Override model replica #%03d on device %d\n", id, p->replicas[id]->dev);

		/* Redirect CUDA calls to model device */
		checkCudaErrors (cudaSetDevice(p->replicas[id]->dev));

		prefix = crossbowStringConcat ("%s/gpu-%02d-replica-%03d", dir, p->replicas[id]->dev, id);
		crossbowModelLoad (p->replicas[id], prefix);
		crossbowStringFree (prefix);
	}

	return 0;
}

int crossbowModelManagerStore (crossbowModelManagerP p, const char *dir) {

	int id;
	crossbowModelP model;
	char *prefix;

	/* Store base models */
	for (id = 0; id < crossbowArrayListSize (p->baseModels); ++id) {

		model = (crossbowModelP) crossbowArrayListGet (p->baseModels, id);
		if (! model)
			continue;

		dbg("Checkpoint base model #%d on device %d\n", id, model->dev);

		/* Redirect CUDA calls to model device */
		checkCudaErrors (cudaSetDevice(model->dev));

		prefix = crossbowStringConcat ("%s/gpu-%02d-theModel", dir, model->dev);
		crossbowModelStore (model, prefix);
		crossbowStringFree (prefix);
	}

	/* Store model replicas */
	for (id = 0; id < p->size; ++id) {

		dbg("Checkpoint model replica #%03d on device %d\n", id, p->replicas[id]->dev);

		/* Redirect CUDA calls to model device */
		checkCudaErrors (cudaSetDevice(p->replicas[id]->dev));

		prefix = crossbowStringConcat ("%s/gpu-%02d-replica-%03d", dir, p->replicas[id]->dev, id);
		crossbowModelStore (p->replicas[id], prefix);
		crossbowStringFree (prefix);
	}

	return 0;
}

/*
 * Notes
 *
 * What is the status of the system when we try to add (respectively, delete) a model replica per device?
 *
 * 1. All replicas are locked since we are in the middle
 *    of synchronisation (`p->locked`).
 *
 * 2. Synchronisation tasks have been scheduled on their
 *    respective devices.
 *
 * 3. The GPU worker attempts to update the clock of one
 *    replica in order to schedule the next task.
 *
 * 4. No thread is trying to modify (dequeue or enqueue)
 *    the queue of available replicas (`p->replicas`).
 */
int crossbowModelManagerAddModel (crossbowModelManagerP p) {

	int dev, id, next;
	int count;
	void *ptr;
	/* char *str; */

	/* New values */
	int size_;
	crossbowModelP *replicas_;
	int *locked_;

	nullPointerException (p);

	/* Lock the model manager */
	pthread_mutex_lock(&(p->lock));

	/* Find number of active devices (based on the number of the base models instantiated) */
	count = 0;
	for (dev = 0; dev < crossbowArrayListSize (p->baseModels); ++dev) {
		ptr = crossbowArrayListGet (p->baseModels, dev);
		if (ptr)
			count ++;
	}
	invalidConditionException (count > 0);
	/* In a homogeneous system, each device has the same number of model replicas */
	invalidConditionException ((p->size % count) == 0);
	dbg("Add %d new replicas\n", count);

	/* Re-allocate arrays and copy existing content */
    size_ = p->size + count;

	locked_ = (int *) crossbowMalloc (size_ * sizeof(int));
	for (id = 0; id < size_; ++id)
		locked_[id] = (id < p->size) ? p->locked[id] : 0;

	replicas_ = (crossbowModelP *) crossbowMalloc (size_ * sizeof(crossbowModelP));
	for (id = 0; id < size_; ++id)
		replicas_[id] = (id < p->size) ? p->replicas[id] : NULL;

	/* Add `count` new model replicas. The new model replicas identifiers start from `next = p->size`. */
	next = p->size;

	for (dev = 0; dev < crossbowArrayListSize (p->baseModels); ++dev) {
		ptr = crossbowArrayListGet (p->baseModels, dev);
		if (! ptr)
			continue;

		/*
		 * Redirect calls to device `dev` and block until all its operations have completed.
		 * The new replica can be a copy of that device's base model or a copy of one of its
		 * replicas.
		 *
		 * We choose to create a copy from the first available replica on the device.
		 */
		checkCudaErrors (cudaSetDevice (dev));
		checkCudaErrors (cudaDeviceSynchronize ());

		/* Find first available replica `dev` (unlikely that there is none). */
		for (id = 0; id < p->size; ++id)
			if (p->replicas[id]->dev == dev)
				break;

		invalidConditionException (id < p->size);

		/* Create a new replica */
		replicas_[next] = crossbowModelReplicate (p->replicas[id], dev, 0); /* Alternatively, base model is `(crossbowModelP) ptr` */

		checkCudaErrors (cudaDeviceSynchronize ());

		/* Store identifier */
		replicas_[next]->id = next;

		/* Lock the new replica */
		crossbowModelLock(replicas_[next]);
		locked_[next] = 1;

		dbg("Created new model replica #%d\n", replicas_[next]->id);

		/* Extend queue with new replica */
		crossbowThetaQueueExpand (p->queue, &(replicas_[next]->id));

		next++;
	}
	/* Done */
	invalidConditionException (next == size_);

	/* Update model manager's state */
	crossbowFree(p->locked, p->size * sizeof(int));
	p->locked = locked_;

	crossbowFree(p->replicas, p->size * sizeof(crossbowModelP));
	p->replicas = replicas_;

	p->size = size_;
    
    /*
#ifdef GPU_VERBOSE
    str = crossbowThreadSafeQueueString (p->queue, INTPTR);
	dbg("%s\n", str);
	crossbowStringFree (str);
#else
    (void) str;
#endif
    */
    
    /* Unlock the model manager */
    pthread_mutex_unlock(&(p->lock));

	return 0;
}

int crossbowModelManagerDelModel (crossbowModelManagerP p) {
	int dev, id, next;
	int count;
	void *ptr;
	char *str;
	/* New values */
	int size_;
	crossbowModelP *replicas_;
	int *locked_;

	nullPointerException (p);

	/* Lock the model manager */
	pthread_mutex_lock(&(p->lock));

	/* Find number of active devices (based on the number of the base models instantiated) */
	count = 0;
	for (dev = 0; dev < crossbowArrayListSize (p->baseModels); ++dev) {
		ptr = crossbowArrayListGet (p->baseModels, dev);
		if (ptr)
			count ++;
	}
	invalidConditionException (count > 0);
	/* In a homogeneous system, each device has the same number of model replicas */
	invalidConditionException ((p->size % count) == 0);
	dbg("Delete %d replicas\n", count);

#ifdef GPU_VERBOSE
	str = crossbowThreadSafeQueueString (p->queue, INTPTR);
	info("%s\n", str);
	crossbowStringFree (str);
#else
	(void) str;
#endif

	/* Re-allocate arrays and copy existing content (note that we are down-sizing the arrays) */
	size_ = p->size - count;
	invalidConditionException (size_ > 0);

	locked_ = (int *) crossbowMalloc (size_ * sizeof(int));
	for (id = 0; id < size_; ++id)
		locked_[id] = p->locked[id];

	replicas_ = (crossbowModelP *) crossbowMalloc (size_ * sizeof(crossbowModelP));
	for (id = 0; id < size_; ++id)
		replicas_[id] = p->replicas[id];

	/* Remove `count` model replicas. The new model replicas identifiers start from `next = size`. */
	next = size_;

	for (dev = 0; dev < crossbowArrayListSize (p->baseModels); ++dev) {
		ptr = crossbowArrayListGet (p->baseModels, dev);
		if (! ptr)
			continue;

		invalidConditionException (dev == p->replicas[next]->dev);

		checkCudaErrors (cudaSetDevice (dev));
		checkCudaErrors (cudaDeviceSynchronize ());

		/* Remove replica from queue */
		dbg("Delete model replica #%d\n", p->replicas[next]->id);

		if (! crossbowThetaQueueShrink (p->queue, &(p->replicas[next]->id)))
			warn("Replica #%d not found\n", p->replicas[next]->id);

		crossbowModelFree (p->replicas[next]);

		next++;
	}
	/* Done */
	invalidConditionException (next == p->size);

	/* Update model manager's state */
	crossbowFree(p->locked, p->size * sizeof(int));
	p->locked = locked_;

	crossbowFree(p->replicas, p->size * sizeof(crossbowModelP));
	p->replicas = replicas_;

	p->size = size_;

#ifdef GPU_VERBOSE
	str = crossbowThreadSafeQueueString (p->queue, INTPTR);
	dbg("%s\n", str);
	crossbowStringFree (str);
#else
	(void) str;
#endif
	/* Unlock the model manager */
	pthread_mutex_unlock(&(p->lock));

	return 0;
}


int crossbowModelManagerDisableModels  (crossbowModelManagerP p) {
	int result = 0;
	if (! p->disabled) {
		result = crossbowThetaQueueDisableAny (p->queue);
		p->disabled = 1;
	}
	return result;
}

void crossbowModelManagerFree (JNIEnv *env, crossbowModelManagerP p) {
	int i;
	int size;
	crossbowModelP model;
	if (! p)
		return;

	crossbowThetaQueueFree (p->queue);

	/* Free base models (incl. theModel) */
	size = crossbowArrayListSize (p->baseModels);
	for (i = 0; i < size; ++i) {
		model = (crossbowModelP) crossbowArrayListGet (p->baseModels, i);
		if (model)
			crossbowModelFree (model);
	}
	crossbowArrayListFree (p->baseModels);
	/* Free multi-GPU synchronisation events */
	crossbowFree (p->synched, (size * sizeof(cudaEvent_t)));
	/* Free replicas */
	for (i = 0; i < p->size; ++i)
		if (p->replicas[i])
			crossbowModelFree (p->replicas[i]);
	/* Unpin class references */
	if (env) {
		(*env)->DeleteGlobalRef(env, p->integerClassRef);
		(*env)->DeleteGlobalRef(env, p->threadClassRef);
	}
	/* Free object */
	crossbowFree(p->locked, p->size * sizeof(int));
	crossbowFree(p->replicas, p->size * sizeof(crossbowModelP));
	crossbowFree(p, sizeof(crossbow_modelmanager_t));
	return;
}

