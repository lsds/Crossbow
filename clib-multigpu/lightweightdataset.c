#include "uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "lightweightdatasetmanager.h"
#include "lightweightdatasethandler.h"

#include "lightweightdatasetprocessor.h"

#include "datasetfile.h"

#include "arraylist.h"

#include <stdlib.h>
#include <limits.h>

#define PHASES 2
#define  TYPES 2

static crossbowLightWeightDatasetManagerP manager [PHASES][TYPES]; /* NULL by default */

static crossbowArrayListP processors = NULL;

static crossbowLightWeightDatasetHandlerP handler = NULL;

static void schedule (crossbowLightWeightDatasetOp_t op, int phi, int slot) {

	crossbowLightWeightDatasetProcessorP processor;
	crossbowLightWeightDatasetProcessorTaskP  task;

	int i;

	/* Get next processor safely (in round-robin fashion) */
	processor = (crossbowLightWeightDatasetProcessorP) crossbowArrayListGetNextSafely (processors);

	/* Get a free task */
	task = crossbowLightWeightDatasetProcessorGetTask (processor);
	invalidConditionException((task->slot[0] == NULL) && (task->slot[1] == NULL));

	/* Initialise it */
	task->op = op;

	task->phi = phi;

	invalidConditionException (manager[phi][0]->GPU == manager[phi][1]->GPU);
	task->GPU = manager[phi][0]->GPU;

	for (i = 0; i < TYPES; ++i) {
		task->slot[i] = &(manager[phi][i]->slots[slot]);
	}

	/* Set handler so that we can reserve/release slot upon task completion */
	task->handler = handler;

	/* Schedule task */
	crossbowLightWeightDatasetProcessorPublish (processor, task);

	return;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_init__II
	(JNIEnv *env, jobject obj, jint numberofprocessors, jint offset) {

	(void) env;
	(void) obj;

#ifdef __LAZY_MAPPING
	illegalStateException ();
#endif
	crossbowMemoryManagerInit ();

	/* Create handler */
	handler = crossbowLightWeightDatasetHandlerCreate (64);

	processors = crossbowArrayListCreate (numberofprocessors);
	int ndx;
	for (ndx = 0; ndx < numberofprocessors; ++ndx)
		crossbowArrayListSet (processors, ndx, crossbowLightWeightDatasetProcessorCreate(offset));

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_init__IIZ_3I
	(JNIEnv *env, jobject obj, jint phase, jint parts, jboolean gpu, jintArray tasksize) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, tasksize);
	jint *argv = (*env)->GetIntArrayElements(env, tasksize, 0);

	invalidArgumentException(argc == PHASES);
	indexOutOfBoundsException(phase, PHASES);

	/* Create a file manager for examples and labels, respectively */
	int i;
	for (i = 0; i < TYPES; i++)
		manager[phase][i] = crossbowLightWeightDatasetManagerCreate (parts, (gpu == JNI_TRUE) ? 1 : 0, argv[i]);

	(*env)->ReleaseIntArrayElements (env, tasksize, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_setPadding
	(JNIEnv *env, jobject obj, jint phase, jintArray padding) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, padding);
	jint *argv = (*env)->GetIntArrayElements(env, padding, 0);

	invalidArgumentException(argc == PHASES);
	indexOutOfBoundsException(phase, PHASES);

	int i;
	for (i = 0; i < TYPES; i++)
		manager[phase][i]->padding = argv[i];

	(*env)->ReleaseIntArrayElements (env, padding, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_setHandler
	(JNIEnv *env, jobject obj, jint phase, jobject buffer, jint count) {

	(void) obj;

	indexOutOfBoundsException (phase, PHASES);
	invalidConditionException (count > 0);

	void *slots = (*env)->GetDirectBufferAddress(env, buffer);

	nullPointerException (handler);
	crossbowLightWeightDatasetHandlerSet (handler, phase, slots, count);

	int i;
	for (i = 0; i < TYPES; i++) {
		manager[phase][i]->buffer = crossbowLightWeightDatasetBufferCreate (count * manager[phase][i]->blocksize);
		/* Lock and register the buffer */
		if (manager[phase][i]->GPU)
			crossbowLightWeightDatasetBufferRegister   (manager[phase][i]->buffer, manager[phase][i]->blocksize);
		crossbowLightWeightDatasetBufferAdviceWillNeed (manager[phase][i]->buffer);
	}

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_configure
	(JNIEnv *env, jobject obj, jint phase, jint batchsize, jint elements, jint fill) {

	(void) env;
	(void) obj;

	int i;
	for (i = 0; i < TYPES; i++) {
		crossbowLightWeightDatasetManagerCreateTasks (manager[phase][i], batchsize, elements, fill);
		crossbowLightWeightDatasetManagerCreateSlots (manager[phase][i], crossbowLightWeightDatasetHandlerNumberOfSlots (handler, phase));
#ifdef GPU_VERBOSE
		crossbowLightWeightDatasetManagerDump (manager[phase][i]);
#endif
	}

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_finalise
	(JNIEnv *env, jobject obj, jint phase) {

	(void) env;
	(void) obj;

	indexOutOfBoundsException (phase, PHASES);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_free
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	/* Free managers */
	int i, j;
	for (i = 0; i < PHASES; ++i) {
		for (j = 0; j < TYPES; ++j) {
#ifdef GPU_VERBOSE
			if (manager[i][j])
				info("file manager %d:%d is %p\n", i, j, manager[i][j]);
			else
				info("file manager %d:%d is null\n", i, j);
#endif
			crossbowLightWeightDatasetManagerFree (manager[i][j]);
		}
	}
	/* Free processors */
	crossbowLightWeightDatasetProcessorP processor;
	int idx;
	for (idx = 0; idx < crossbowArrayListSize (processors); idx++) {
		processor = (crossbowLightWeightDatasetProcessorP) crossbowArrayListGet (processors, idx);
		crossbowLightWeightDatasetProcessorFree (processor);
	}
	crossbowArrayListFree (processors);

	/* Free handler */
	crossbowLightWeightDatasetHandlerFree (handler);

	crossbowMemoryManagerDump ();

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_register
	(JNIEnv *env, jobject obj, jint phase, jint type, jint id, jstring filename) {
	(void) obj;

	indexOutOfBoundsException(phase, PHASES);
	indexOutOfBoundsException(type,   TYPES);

	const char *binding = (*env)->GetStringUTFChars (env, filename, NULL);

	crossbowLightWeightDatasetManagerRegister (manager[phase][type], id, binding);

	(*env)->ReleaseStringUTFChars (env, filename, binding);

	return 0;
}

JNIEXPORT jlong JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_address
	(JNIEnv *env, jobject obj, jint phase, jint type) {

	(void) env;
	(void) obj;

	indexOutOfBoundsException(phase, PHASES);
	indexOutOfBoundsException(type,   TYPES);

	return (jlong) crossbowLightWeightDatasetBufferAddress (manager[phase][type]->buffer);
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_capacity
	(JNIEnv *env, jobject obj, jint phase, jint type) {

	(void) env;
	(void) obj;

	indexOutOfBoundsException(phase, PHASES);
	indexOutOfBoundsException(type,   TYPES);

	return crossbowLightWeightDatasetBufferCapacity (manager[phase][type]->buffer);
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_reserve
	(JNIEnv *env, jobject obj, jint phase, jlongArray freeP) {

	(void) obj;

	jsize  argc = (*env)->GetArrayLength(env, freeP);
	jlong *argv = (*env)->GetLongArrayElements(env, freeP, 0);

	invalidArgumentException(argc == PHASES);
	indexOutOfBoundsException(phase, PHASES);

	/* Find the slot index */
	int slot = crossbowLightWeightDatasetHandlerTranslate (argv[0], manager[phase][0]->blocksize);

	crossbowLightWeightDatasetHandlerPrepareToReserve (handler, phase, slot);

	dbg("Reserve slot %03d (free %010ld and/or %010ld); next task is %06d\n", slot, argv[0], argv[1], manager[phase][0]->slots[slot].ndx);

	/* Schedule a data set task */
	schedule (RESERVE, phase, slot);

	(*env)->ReleaseLongArrayElements (env, freeP, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_LightWeightDatasetMemoryManager_release
	(JNIEnv *env, jobject obj, jint phase, jlong freeP) {

	(void) env;
	(void) obj;

	indexOutOfBoundsException(phase, PHASES);

	/* Find the slot index */
	int slot = crossbowLightWeightDatasetHandlerTranslate (freeP, manager[phase][0]->blocksize);

	dbg("Release slot %03d (free %010ld)\n", slot, freeP);

	crossbowLightWeightDatasetHandlerPrepareToRelease (handler, phase, slot);

	/* Schedule a data set task */
	schedule (RELEASE, phase, slot);

	return 0;
}
