#ifndef __CROSSBOW_MODELMANAGER_H_
#define __CROSSBOW_MODELMANAGER_H_

#include "model.h"
#include "thetaqueue.h"

#include "arraylist.h"

#include "utils.h"

#include <jni.h>

#include <cuda.h>

typedef struct crossbow_modelmanager *crossbowModelManagerP;
typedef struct crossbow_modelmanager {

	crossbowModelP theModel;

	crossbowArrayListP baseModels;
	cudaEvent_t *synched;

	int size;
	crossbowModelP *replicas;

	jclass integerClassRef;
	jmethodID constructor, intValue;

	jclass threadClassRef;
	jmethodID yield;

	crossbowThetaQueueP queue;

	crossbowModelSynchronisation_t type;

	/* Internal state per model replica: if `locked[id]` is not 0 then model replica `id` is locked */
	int *locked;
	
	int disabled;

	pthread_mutex_t lock;

} crossbow_modelmanager_t;

crossbowModelManagerP crossbowModelManagerCreate (JNIEnv *, int, crossbowModelP, crossbowModelSynchronisation_t, crossbowArrayListP);

crossbowModelP crossbowModelManagerGetNextOrWait (JNIEnv *, crossbowModelManagerP, int);

crossbowModelP crossbowModelManagerGet (JNIEnv *, crossbowModelManagerP, jobject);

jobject crossbowModelManagerAcquireAccess (JNIEnv *, crossbowModelManagerP, int *);

jobject crossbowModelManagerUpgradeAccess (JNIEnv *, crossbowModelManagerP, jobject, int *);

int crossbowModelManagerLockAll (crossbowModelManagerP);

int crossbowModelManagerLockAny (crossbowModelManagerP);

int crossbowModelManagerUnlockAny (crossbowModelManagerP);

int crossbowModelManagerNumberOfVisibleUpdates (crossbowModelManagerP);

void crossbowModelManagerIncrementClock (crossbowModelManagerP, int);

void crossbowModelManagerRelease (crossbowModelManagerP, crossbowModelP);

int crossbowModelManagerStore (crossbowModelManagerP, const char *);

int crossbowModelManagerLoad  (crossbowModelManagerP, const char *);

int crossbowModelManagerAddModel  (crossbowModelManagerP);

int crossbowModelManagerDelModel  (crossbowModelManagerP);

int crossbowModelManagerDisableModels  (crossbowModelManagerP);

void crossbowModelManagerFree (JNIEnv *, crossbowModelManagerP);

#endif /* __CROSSBOW_MODELMANAGER_H_ */
