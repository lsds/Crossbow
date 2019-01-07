#include "uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "datasetfilemanager.h"
#include "datasetfilehandler.h"

#include "datasetfile.h"

#include "arraylist.h"

#include "memoryregionpool.h"

#include <stdlib.h>
#include <limits.h>

#define PHASES 2
#define  TYPES 2

static crossbowDatasetFileManagerP filemanager [PHASES][TYPES]; /* NULL by default */

static crossbowArrayListP datasetfilehandlers = NULL;

static void slide (int phase, int fid, unsigned op) {

	indexOutOfBoundsException(phase, PHASES);

	crossbowMemoryRegionP region;
	crossbowMemoryRegistryNodeP node;
	crossbowDatasetFileHandlerP handler;
	crossbowDatasetFileBlockP p;

	int blocks;
	int i, j;
	int offset;
	for (i = 0; i < TYPES; ++i) {
		/* Get data set file node  */
		node = crossbowMemoryRegistryGet (filemanager[phase][i]->registry, fid);
		dbg ("Slide %3s file %s\n", (op == 0 ? "out" : "in"), node->file->filename);
#ifdef __LAZY_MAPPING
		/* Ensure that the file is mapped when we register it */
		if (op == 1)
			crossbowDatasetFileMap (node->file);
#endif
		/* Split file into blocks; assign to it a memory region if required */
		if (filemanager[phase][i]->copyconstructor) {
			
            /* Do not take into account padding when spliting the file into blocks */
			blocks = node->file->length / (filemanager[phase][i]->blocksize - filemanager[phase][i]->pad);
            
            if (op == 1) {
                
                /* Get a memory region */
                region = crossbowMemoryRegionPoolGet (filemanager[phase][i]->pool);
                
                /* Set the limit for the memory region */
                crossbowMemoryRegionSetLimit (region, blocks * filemanager[phase][i]->blocksize);
				
                /* Associate memory region with the file */
                region->file = node->file;
                info("Associated region #%03d with %s\n", region->id, node->file->filename);
                
                /* Assign memory region to the file */
				crossbowDatasetFileAssign (node->file, (void *) region);
			}
            
            /* Ensure that file has been associated with a memory region */
			invalidConditionException(node->file->region != NULL);
		}
		else {
            /* The original file is already padded */
			blocks = node->file->length /  filemanager[phase][i]->blocksize;
		}
        
		offset = 0;
		for (j = 0; j < blocks; ++j) {
            
			/* Get next handler (in round-robin fashion) */
			handler = (crossbowDatasetFileHandlerP) crossbowArrayListGetNext (datasetfilehandlers);
			
            /* Get a free block */
			p = crossbowDatasetFileHandlerGetBlock (handler);
			invalidConditionException(p->file == NULL);
			
            /* Initialise it */
			p->file = node->file;
			p->offset = offset;
			p->length = filemanager[phase][i]->blocksize;
			p->pad = filemanager[phase][i]->pad;
			p->gpu = filemanager[phase][i]->gpu;
			p->op = op;
			
            /* Schedule task */
			crossbowDatasetFileHandlerPublish (handler, p);
			
            /* Increment offset */
			offset += filemanager[phase][i]->blocksize;
		}
	}
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_init__II
	(JNIEnv *env, jobject obj, jint numberofdatasetfilehandlers, jint offset) {

	(void) env;
	(void) obj;

	crossbowMemoryManagerInit ();

	datasetfilehandlers = crossbowArrayListCreate (numberofdatasetfilehandlers);
	int ndx;
	for (ndx = 0; ndx < numberofdatasetfilehandlers; ++ndx)
		crossbowArrayListSet (datasetfilehandlers, ndx, crossbowDatasetFileHandlerCreate(offset));

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_init__IIZ_3I
	(JNIEnv *env, jobject obj, jint phase, jint parts, jboolean gpu, jintArray tasksize) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, tasksize);
	jint *argv = (*env)->GetIntArrayElements(env, tasksize, 0);

	invalidArgumentException(argc == PHASES);
	indexOutOfBoundsException(phase, PHASES);

	/* Create a file manager for examples and labels, respectively */
	int i;
	for (i = 0; i < 2; i++)
		filemanager[phase][i] = crossbowDatasetFileManagerCreate (parts, (gpu == JNI_TRUE) ? 1 : 0, argv[i]);

	(*env)->ReleaseIntArrayElements (env, tasksize, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_configure__I_3I
	(JNIEnv *env, jobject obj, jint phase, jintArray padding) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, padding);
	jint *argv = (*env)->GetIntArrayElements(env, padding, 0);

	invalidArgumentException(argc == PHASES);
	indexOutOfBoundsException(phase, PHASES);

	int i;
	for (i = 0; i < 2; i++)
		filemanager[phase][i]->pad = argv[i];

	(*env)->ReleaseIntArrayElements (env, padding, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_configure__I_3Z
	(JNIEnv *env, jobject obj, jint phase, jbooleanArray copy) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, copy);
	jboolean *argv = (*env)->GetBooleanArrayElements(env, copy, 0);

	invalidArgumentException(argc == PHASES);
	indexOutOfBoundsException(phase, PHASES);

	/*
	 * We cannot create a pool of temporary memory regions yet because we don't the maximum file size.
	 * We should wait until the call to `finalise`.
	 */
	int i;
	for (i = 0; i < 2; i++)
		filemanager[phase][i]->copyconstructor = (argv[i] == JNI_TRUE) ? 1 : 0;

	(*env)->ReleaseBooleanArrayElements (env, copy, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_finalise
	(JNIEnv *env, jobject obj, jint phase) {

	(void) env;
	(void) obj;

	indexOutOfBoundsException(phase, PHASES);

	int i;
	int capacity;
	crossbowMemoryRegistryNodeP node;
	int id;

	for (i = 0; i < 2; i++) {
		if (! filemanager[phase][i]->copyconstructor)
			continue;
		/* Find maximum memory region capacity */
		capacity = 0;
		for (id = 0; id < crossbowMemoryRegistrySize (filemanager[phase][i]->registry); ++id) {
			node = crossbowMemoryRegistryGet (filemanager[phase][i]->registry, id);
			if (capacity < crossbowDatasetFileSize(node->file))
				capacity = crossbowDatasetFileSize(node->file);
		}
		/* Add padding */
		capacity += ((capacity / (filemanager[phase][i]->blocksize - filemanager[phase][i]->pad)) * filemanager[phase][i]->pad);
		filemanager[phase][i]->pool = crossbowMemoryRegionPoolCreate (capacity);
	}

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_free
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	int i, j;
	for (i = 0; i < PHASES; ++i) {
		for (j = 0; j < TYPES; ++j) {
#ifdef GPU_VERBOSE
			if (filemanager[i][j])
				info("file manager %d:%d is %p\n", i, j, filemanager[i][j]);
			else
				info("file manager %d:%d is null\n", i, j);
#endif
			crossbowDatasetFileManagerFree (filemanager[i][j]);
		}
	}
	/* Free file handlers */
	crossbowDatasetFileHandlerP handler;
	int idx;
	for (idx = 0; idx < crossbowArrayListSize (datasetfilehandlers); idx++) {
		handler = (crossbowDatasetFileHandlerP) crossbowArrayListGet (datasetfilehandlers, idx);
		crossbowDatasetFileHandlerFree (handler);
	}
	crossbowArrayListFree (datasetfilehandlers);

	crossbowMemoryManagerDump ();

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_register
	(JNIEnv *env, jobject obj, jint phase, jint type, jint id, jstring filename) {

	(void) obj;

	indexOutOfBoundsException(phase, PHASES);
	indexOutOfBoundsException(type,   TYPES);

	const char *binding = (*env)->GetStringUTFChars (env, filename, NULL);

	crossbowDatasetFileManagerRegister (filemanager[phase][type], id, binding);

	(*env)->ReleaseStringUTFChars (env, filename, binding);

	return 0;
}

JNIEXPORT jlong JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_address
	(JNIEnv *env, jobject obj, jint phase, jint type, jint id) {

	(void) env;
	(void) obj;

	indexOutOfBoundsException(phase, PHASES);
	indexOutOfBoundsException(type,   TYPES);

	crossbowMemoryRegistryNodeP p = crossbowMemoryRegistryGet (filemanager[phase][type]->registry, id);
	if (p->file->region != NULL)
		return (jlong) crossbowMemoryRegionAddress (p->file->region);
	else
		return (jlong) crossbowDatasetFileAddress (p->file);
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_capacity
	(JNIEnv *env, jobject obj, jint phase, jint type, jint id) {

	(void) env;
	(void) obj;

	long tasksize, filesize;
	long bytes;

	indexOutOfBoundsException(phase, PHASES);
	indexOutOfBoundsException(type,   TYPES);

	crossbowMemoryRegistryNodeP p = crossbowMemoryRegistryGet (filemanager[phase][type]->registry, id);
	/*
	 * If we are using a copy constructor, we need to account for
	 * the padding added to each block for page alignment.
	 */
	if (filemanager[phase][type]->copyconstructor) {

		tasksize = filemanager[phase][type]->blocksize - filemanager[phase][type]->pad;
		filesize = crossbowDatasetFileSize(p->file);

		/* Assert than file size is a multiple of the non-padded task size */
		if ((filesize % tasksize) != 0) {
			err("File's %s size (%ld bytes) is not a multiple of %ld\n", p->file->filename, filesize, tasksize);
		}
		invalidConditionException((filesize % tasksize) == 0);


		bytes = filesize + ((filesize / tasksize) * filemanager[phase][type]->pad);

		/* Assert than bytes is less than INT_MAX */
		invalidConditionException(bytes < INT_MAX);

		return bytes;
	}
	else {
		return crossbowDatasetFileSize(p->file);
	}
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_slideOut
	(JNIEnv *env, jobject obj, jint phase, jint fid) {

	(void) env;
	(void) obj;

	slide (phase, fid, 0);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_dataset_DatasetMemoryManager_slideIn
	(JNIEnv *env, jobject obj, jint phase, jint fid) {

	(void) env;
	(void) obj;

	slide (phase, fid, 1);

	return 0;
}
