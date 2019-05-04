#include "uk_ac_imperial_lsds_crossbow_device_TheGPU.h"

#include <jni.h>

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "executioncontext.h"

static crossbowExecutionContextP theGPU = NULL;

static jclass threadClassRef;
static jmethodID yield;

static jclass    mappedDataByteBufferClassRef  = NULL;
static jfieldID  mappedDataBufferAddressField  = NULL;
static jfieldID  mappedDataBufferCapacityField = NULL;

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_init
	(JNIEnv *env, jobject obj, jintArray devices, jint numberofstreams,
			jint numberofcallbackhandlers, jint numberoftaskhandlers,
			jint callbackthreadcoreoffset, jint taskthreadcoreoffset) {

	(void) obj;

	if (theGPU) {
		fprintf(stderr, "error: GPU execution context already initialised\n");
		exit(1);
	}

	jsize argc = (*env)->GetArrayLength(env, devices);
	jint *argv = (*env)->GetIntArrayElements(env, devices, 0);

	int *offset = (int *) malloc (sizeof(int) + sizeof(int));
	if (! offset) {
		fprintf(stderr, "fatal error: out of memory\n");
		exit(1);
	}
	offset[0] = callbackthreadcoreoffset;
	offset[1] =     taskthreadcoreoffset;

	theGPU = crossbowExecutionContextInit (argv, argc, 
		numberofstreams, numberofcallbackhandlers, numberoftaskhandlers, offset);

	(*env)->ReleaseIntArrayElements (env, devices, argv, JNI_ABORT);

	free (offset);

	/* Static references to Java Thread class */
	threadClassRef = (jclass) (*env)->NewGlobalRef (env, (*env)->FindClass (env, "java/lang/Thread"));
	yield = (*env)->GetStaticMethodID (env, threadClassRef, "yield", "()V");

	/* Static references to Crossbow's MappedDataBuffer class */
	mappedDataByteBufferClassRef  = (jclass) (*env)->NewGlobalRef
			(env, (*env)->FindClass (env, "uk/ac/imperial/lsds/crossbow/data/MappedDataBuffer"));

	mappedDataBufferAddressField  = (*env)->GetFieldID(env, mappedDataByteBufferClassRef, "address",  "J");
	mappedDataBufferCapacityField = (*env)->GetFieldID(env, mappedDataByteBufferClassRef, "capacity", "I");

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_free
	(JNIEnv *env, jobject obj) {

	(void) obj;

	crossbowExecutionContextFree (env, theGPU);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_dump
	(JNIEnv *env, jobject obj) {
	
	(void) env;
	(void) obj;
	
	crossbowExecutionContextDump (theGPU);

        return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchExamples
	(JNIEnv *env, jobject obj, jintArray shape, jint size) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, shape);
	jint *argv = (*env)->GetIntArrayElements(env, shape, 0);

	crossbowExecutionContextSetBatchExamples (theGPU, argc, argv, size);

	(*env)->ReleaseIntArrayElements (env, shape, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchLabels
	(JNIEnv *env, jobject obj, jintArray shape, jint size) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, shape);
	jint *argv = (*env)->GetIntArrayElements(env, shape, 0);

	crossbowExecutionContextSetBatchLabels (theGPU, argc, argv, size);

	(*env)->ReleaseIntArrayElements (env, shape, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchSplits
	(JNIEnv *env, jobject obj, jint splits) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetBatchSplits (theGPU, splits);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureStreams
	(JNIEnv *env, jobject obj, int branches) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCreateStreams (theGPU, branches);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setRandomSeed
	(JNIEnv *env, jobject obj, jlong seed) {
	
	(void) env;
	(void) obj;

	crossbowExecutionContextSetRandomSeed (theGPU, (unsigned long long) seed);
	
	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernel
	(JNIEnv *env, jobject obj, jint id, jstring name, jint inputs, jint variables, jint outputs, jboolean pull) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernel (theGPU, id, binding, inputs, variables, outputs, (pull == JNI_TRUE) ? 1 : 0);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

/* A kernel has one or more inputs, identified by `ndx` */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelInput
	(JNIEnv *env, jobject obj, jint id, jint ndx, jintArray shape, jint capacity) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, shape);
	jint *argv = (*env)->GetIntArrayElements(env, shape, 0);

	crossbowExecutionContextSetKernelInput (theGPU, id, ndx, argc, argv, capacity);

	(*env)->ReleaseIntArrayElements (env, shape, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelOutput
	(JNIEnv *env, jobject obj, jint id, jintArray shape, jint capacity) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, shape);
	jint *argv = (*env)->GetIntArrayElements(env, shape, 0);

	crossbowExecutionContextSetKernelOutput (theGPU, id, argc, argv, capacity);

	(*env)->ReleaseIntArrayElements (env, shape, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelLocalVariable
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jintArray shape, jint capacity, jboolean readonly) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, shape);
	jint *argv = (*env)->GetIntArrayElements(env, shape, 0);

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelLocalVariable (theGPU, id, ndx, binding, argc, argv, capacity, readonly);

	(*env)->ReleaseIntArrayElements (env, shape, argv, JNI_ABORT);
	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelLocalVariableBuffer
	(JNIEnv *env, jobject obj, jint id, jint ndx, jobject buffer) {

	(void) obj;

	void *src = (*env)->GetDirectBufferAddress(env, buffer);

	crossbowExecutionContextSetKernelLocalVariableBuffer (theGPU, id, ndx, src);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalars
	(JNIEnv *env, jobject obj, jint id, jint count) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetKernelScalars (theGPU, id, count);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsInt
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jint value) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelScalarAsInt (theGPU, id, ndx, binding, value);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsFloat
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jfloat value) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelScalarAsFloat (theGPU, id, ndx, binding, value);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsDouble
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jdouble value) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelScalarAsDouble (theGPU, id, ndx, binding, value);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelType
	(JNIEnv *env, jobject obj, jint id, jint type) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetKernelType (theGPU, id, type);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelInputDescriptor
	(JNIEnv *env, jobject obj, jint id, jint count, jint channels, jint height, jint width) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetKernelInputDescriptor (theGPU, id, count, channels, height, width);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelOutputDescriptor
	(JNIEnv *env, jobject obj, jint id, jint count, jint channels, jint height, jint width) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetKernelOutputDescriptor (theGPU, id, count, channels, height, width);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionDescriptor
	(JNIEnv *env, jobject obj, jint id, jint paddingHeight, jint paddingWidth, jint strideHeight, jint strideWidth) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetConvolutionDescriptor (theGPU, id, paddingHeight, paddingWidth, strideHeight, strideWidth);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionFilterDescriptor
	(JNIEnv *env, jobject obj, jint id, jint count, jint channels, jint height, jint width) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetConvolutionFilterDescriptor (theGPU, id, count, channels, height, width);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionBiasDescriptor
	(JNIEnv *env, jobject obj, jint id, jint count, jint channels, jint height, jint width) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetConvolutionBiasDescriptor (theGPU, id, count, channels, height, width);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionForwardAlgorithm
	(JNIEnv *env, jobject obj, jint id, jint limit, jdouble threshold) {

	(void) env;
	(void) obj;

	size_t workSpaceSize = crossbowExecutionContextCudnnConfigureConvolutionForwardAlgorithm (theGPU, id, limit, threshold);

	return (int) workSpaceSize;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionBackwardFilterAlgorithm
	(JNIEnv *env, jobject obj, jint id, jint limit, jdouble threshold) {

	(void) env;
	(void) obj;

	size_t workSpaceSize = crossbowExecutionContextCudnnConfigureConvolutionBackwardFilterAlgorithm (theGPU, id, limit, threshold);

	return (int) workSpaceSize;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionBackwardDataAlgorithm
	(JNIEnv *env, jobject obj, jint id, jint limit, jdouble threshold) {

	(void) env;
	(void) obj;

	size_t workSpaceSize = crossbowExecutionContextCudnnConfigureConvolutionBackwardDataAlgorithm (theGPU, id, limit, threshold);

	return (int) workSpaceSize;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetPoolingMode
	(JNIEnv *env, jobject obj, jint id, jint mode) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetPoolingMode (theGPU, id, mode);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetPoolingDescriptor
	(JNIEnv *env, jobject obj, jint id, int windowHeight, jint windowWidth, jint paddingHeight, jint paddingWidth, jint strideHeight, jint strideWidth) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetPoolingDescriptor (theGPU, id, windowHeight, windowWidth, paddingHeight, paddingWidth, strideHeight, strideWidth);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetActivationDescriptor
	(JNIEnv *env, jobject obj, jint id, jint mode, jdouble ceiling) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetActivationDescriptor (theGPU, id, mode, ceiling);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetBatchNormDescriptor
	(JNIEnv *env, jobject obj, jint id) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetBatchNormDescriptor (theGPU, id);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetBatchNormEstimatedMeanAndVariance
	(JNIEnv *env, jobject obj, jint id, jint capacity) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetBatchNormEstimatedMeanAndVariance (theGPU, id, capacity);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetDropoutDescriptor
	(JNIEnv *env, jobject obj, jint id, jfloat dropout, jlong seed) {

	(void) env;
	(void) obj;

	crossbowExecutionContextCudnnSetDropoutDescriptor (theGPU, id, dropout, seed);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnGetDropoutReserveSpaceSize
	(JNIEnv *env, jobject obj, jint id) {

	(void) env;
	(void) obj;

	size_t reserveSpaceSize = crossbowExecutionContextCudnnGetDropoutReserveSpaceSize (theGPU, id);

	return (int) reserveSpaceSize;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameters
	(JNIEnv *env, jobject obj, jint id, jint count) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetKernelConfigurationParameters (theGPU, id, count);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsInt
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jint value) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelConfigurationParameterAsInt (theGPU, id, ndx, binding, value);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsFloat
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jfloat value) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelConfigurationParameterAsFloat (theGPU, id, ndx, binding, value);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}


JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsIntArray
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jintArray values) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	jsize argc = (*env)->GetArrayLength(env, values);
	jint *argv = (*env)->GetIntArrayElements(env, values, 0);

	crossbowExecutionContextSetKernelConfigurationParameterAsIntArray (theGPU, id, ndx, binding, argc, argv);

	(*env)->ReleaseIntArrayElements (env, values, argv, JNI_ABORT);
	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsFloatArray
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jfloatArray values) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	jsize   argc = (*env)->GetArrayLength(env, values);
	jfloat *argv = (*env)->GetFloatArrayElements(env, values, 0);

	crossbowExecutionContextSetKernelConfigurationParameterAsFloatArray (theGPU, id, ndx, binding, argc, argv);

	(*env)->ReleaseFloatArrayElements (env, values, argv, JNI_ABORT);
	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsDouble
	(JNIEnv *env, jobject obj, jint id, jint ndx, jstring name, jdouble value) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, name, NULL);

	crossbowExecutionContextSetKernelConfigurationParameterAsDouble (theGPU, id, ndx, binding, value);

	(*env)->ReleaseStringUTFChars (env, name, binding);

	return 0;
}

/* A kernel is referenced by one of more dataflow nodes (or operators) */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowGraph
	(JNIEnv *env, jobject obj, jint id, jintArray ops) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, ops);
	jint *argv = (*env)->GetIntArrayElements(env, ops, 0);

	crossbowExecutionContextSetDataflowGraph (theGPU, id, argc, argv);

	(*env)->ReleaseIntArrayElements (env, ops, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowStream
	(JNIEnv *env, jobject obj, jint id, jint ord, jint branch) {
	
	(void) env;
	(void) obj;
	
	crossbowExecutionContextSetDataflowStream (theGPU, id, ord, branch);
	
	return 0;
	
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDependency
	(JNIEnv *env, jobject obj, jint id, jint ord, jint type, jint guard, jboolean internal) {
	
	(void) env;
	(void) obj;
	
	crossbowExecutionContextSetDataflowDependency (theGPU, id, ord, type, guard, (internal == JNI_TRUE) ? 0 : 1);
	
	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowUpstreamNeighbours
	(JNIEnv *env, jobject obj, jint id, jint ord, jintArray neighbours) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, neighbours);
	jint *argv = (*env)->GetIntArrayElements(env, neighbours, 0);

	crossbowExecutionContextSetDataflowUpstreamNeighbours (theGPU, id, ord, argc, argv);

	(*env)->ReleaseIntArrayElements (env, neighbours, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDownstreamNeighbours
	(JNIEnv *env, jobject obj, jint id, jint ord, jintArray neighbours) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, neighbours);
	jint *argv = (*env)->GetIntArrayElements(env, neighbours, 0);

	crossbowExecutionContextSetDataflowDownstreamNeighbours (theGPU, id, ord, argc, argv);

	(*env)->ReleaseIntArrayElements (env, neighbours, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowLossOperator
	(JNIEnv *env, jobject obj, jint id, jint op) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetDataflowLossOperator(theGPU, id, op);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowAccuracyOperator
	(JNIEnv *env, jobject obj, jint id, jint op) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetDataflowAccuracyOperator(theGPU, id, op);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDataTransformOperator
	(JNIEnv *env, jobject obj, jint id, jint op) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetDataflowDataTransformOperator(theGPU, id, op);

	return 0;
}

/* A kernel is referenced by one of more dataflow nodes (or operators) */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowPeers
	(JNIEnv *env, jobject obj, jint id, jintArray peers) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, peers);
	jint *argv = (*env)->GetIntArrayElements(env, peers, 0);

	crossbowExecutionContextSetDataflowPeers (theGPU, id, argc, argv);

	(*env)->ReleaseIntArrayElements (env, peers, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowMemoryPlan
	(JNIEnv *env, jobject obj, jint id, jint order, jint provider, jint position) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetDataflowMemoryPlan (theGPU, id, order, provider, position);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModel
	(JNIEnv *env, jobject obj, jint variables, jint size) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetModel (theGPU, variables, size);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariable
	(JNIEnv *env, jobject obj, jint id, jint order, jintArray dims, jint capacity) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, dims);
	jint *argv = (*env)->GetIntArrayElements(env, dims, 0);

	crossbowExecutionContextSetModelVariable (theGPU, id, order, argc, argv, capacity);

	(*env)->ReleaseIntArrayElements (env, dims, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariableBuffer
	(JNIEnv *env, jobject obj, jint id, jint order, jobject buffer) {

	(void) obj;

	void *src = (*env)->GetDirectBufferAddress(env, buffer);

	crossbowExecutionContextSetModelVariableBuffer (theGPU, id, order, src);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariableLearningRateMultiplier
	(JNIEnv *env, jobject obj, jint id, jint order, jfloat multiplier) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetModelVariableLearningRateMultiplier (theGPU, id, order, multiplier);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelWorkPerClock
	(JNIEnv *env, jobject obj, jint wpc) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetModelWorkPerClock (theGPU, wpc);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setUpdateModelType
	(JNIEnv *env, jobject obj, jint type) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetUpdateModelType (theGPU, type);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyFixed
	(JNIEnv *env, jobject obj, jfloat rate) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetLearningRateDecayPolicyFixed (theGPU, rate);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyInv
	(JNIEnv *env, jobject obj, jfloat learningRate, jdouble gamma, jdouble power) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetLearningRateDecayPolicyInv (theGPU, learningRate, gamma, power);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyStep
	(JNIEnv *env, jobject obj, jfloat learningRate, jdouble gamma, jint size) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetLearningRateDecayPolicyStep (theGPU, learningRate, gamma, size);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyMultiStep
	(JNIEnv *env, jobject obj, jfloat learningRate, jdouble gamma, jint warmuptasks, jintArray steps) {

	(void) env;
	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, steps);
	jint *argv = (*env)->GetIntArrayElements(env, steps, 0);

	crossbowExecutionContextSetLearningRateDecayPolicyMultiStep (theGPU, learningRate, gamma, warmuptasks, argc, argv);

	(*env)->ReleaseIntArrayElements (env, steps, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyExp
	(JNIEnv *env, jobject obj, jfloat learningRate, jdouble gamma) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetLearningRateDecayPolicyExp (theGPU, learningRate, gamma);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyCircular
	(JNIEnv *env, jobject obj, jfloatArray rate, jint superconvergence, jfloatArray momentum, jint step) {

	(void) obj;

	jsize argc = (*env)->GetArrayLength(env, rate);

	invalidConditionException(argc == 3);

	jfloat *H = (*env)->GetFloatArrayElements(env,     rate, 0);
	jfloat *M = (*env)->GetFloatArrayElements(env, momentum, 0);

	crossbowExecutionContextSetLearningRateDecayPolicyCircular (theGPU, H, superconvergence, M, step);

	(*env)->ReleaseFloatArrayElements (env,     rate, H, JNI_ABORT);
	(*env)->ReleaseFloatArrayElements (env, momentum, M, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setBaseModelMomentum
	(JNIEnv *env, jobject obj, jfloat momentum) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetBaseModelMomentum (theGPU, momentum);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setMomentum
	(JNIEnv *env, jobject obj, jfloat momentum, jint method) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetMomentum (theGPU, momentum, method);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setWeightDecay
	(JNIEnv *env, jobject obj, jfloat weigthDecay) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetWeightDecay (theGPU, weigthDecay);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setEamsgdAlpha
	(JNIEnv *env, jobject obj, jfloat alpha) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetEamsgdAlpha (theGPU, alpha);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setEamsgdTau
	(JNIEnv *env, jobject obj, jint tau) {

	(void) env;
	(void) obj;

	crossbowExecutionContextSetEamsgdTau (theGPU, tau);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelManager
	(JNIEnv *env, jobject obj, jint replicas, jint type) {

	(void) obj;

	crossbowExecutionContextSetModelManager (env, theGPU, replicas, type);

	return 0;
}

JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_acquireAccess
	(JNIEnv *env, jobject obj, jintArray clock) {

	(void) obj;

	jobject result = NULL;

	jsize argc = (*env)->GetArrayLength(env, clock);
	jint *argv = (*env)->GetIntArrayElements(env, clock, 0);
	invalidArgumentException (argc == 1);

	result = crossbowExecutionContextAcquireAccess (env, theGPU, &argv[0]);

	(*env)->ReleaseIntArrayElements (env, clock, argv, JNI_COMMIT);
	return result;
}

JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_upgradeAccess
	(JNIEnv *env, jobject obj, jobject replicaId, jintArray clock) {

	(void) obj;

	jobject result = NULL;

	jsize argc = (*env)->GetArrayLength(env, clock);
	jint *argv = (*env)->GetIntArrayElements(env, clock, 0);
	invalidArgumentException (argc == 1);

	result = crossbowExecutionContextUpgradeAccess (env, theGPU, replicaId, &argv[0]);

	(*env)->ReleaseIntArrayElements (env, clock, argv, JNI_COMMIT);

	return result;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_release
	(JNIEnv *env, jobject obj, jobject replicaId) {

	(void) env;
	(void) obj;

	(void) replicaId;

	err("Cannot release a GPU model replica id object from the GPU\n");
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setResultHandler
	(JNIEnv *env, jobject obj, jint id, jobject buffer, jint count) {

	(void) obj;

	void *slots = (*env)->GetDirectBufferAddress(env, buffer);

	crossbowExecutionContextSetResultHandler (theGPU, id, slots, count);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLightWeightDatasetHandler
	(JNIEnv *env, jobject obj, jint id, jobject buffer, jint count) {

	(void) obj;

	void *slots = (*env)->GetDirectBufferAddress(env, buffer);

	crossbowExecutionContextSetLightWeightDatasetHandler (theGPU, id, slots, count);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_execute
	(JNIEnv *env, jobject obj,
	jint dataflowId,
	jint taskId,
	jobject examples, jint examplesStartP, jint examplesEndP,
	jobject   labels, jint   labelsStartP, jint   labelsEndP,
	jlongArray freeP,
	jint phase,
	jobject replica) {

	(void) obj;

	/*
	 * We need custom functions to get the data pointer and capacity
	 * of a MappedDataBuffer.
	 *
	 * void *examplesP = (*env)->GetDirectBufferAddress(env, examples);
	 * void *labelsP   = (*env)->GetDirectBufferAddress(env,   labels);
	 *
	 * int examplesCapacity = (*env)->GetDirectBufferCapacity(env, examples);
	 * int labelsCapacity   = (*env)->GetDirectBufferCapacity(env,   labels);
	 */

	void *examplesP = (void *) (intptr_t) (*env)->GetLongField (env, examples, mappedDataBufferAddressField);
	void *labelsP   = (void *) (intptr_t) (*env)->GetLongField (env, labels,   mappedDataBufferAddressField);

	int examplesCapacity = (*env)->GetIntField(env, examples, mappedDataBufferCapacityField);
	int   labelsCapacity = (*env)->GetIntField(env, labels,   mappedDataBufferCapacityField);

	/* Get free pointers (int free [2]) */
	jsize  argc = (*env)->GetArrayLength(env, freeP);
	jlong *argv = (*env)->GetLongArrayElements(env, freeP, 0);
	invalidArgumentException (argc == 2);

	crossbowExecutionContextExecute (
		env,
		theGPU,
		dataflowId,
		taskId,
		examplesP, examplesCapacity, examplesStartP, examplesEndP,
		labelsP,     labelsCapacity,   labelsStartP,   labelsEndP,
		argv,
		phase,
		replica);

	(*env)->ReleaseLongArrayElements (env, freeP, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_schedule
	(JNIEnv *env, jobject obj,
	jint dataflowId,
	jint taskId,
	jobject examples, jlong examplesStartP, jlong examplesEndP,
	jobject   labels, jlong   labelsStartP, jlong   labelsEndP,
	jlongArray freeP,
	jint phase,
	jint bound) {

	(void) obj;

	void *examplesP = (void *) (intptr_t) (*env)->GetLongField (env, examples, mappedDataBufferAddressField);
	void *labelsP   = (void *) (intptr_t) (*env)->GetLongField (env, labels,   mappedDataBufferAddressField);

	int examplesCapacity = (*env)->GetIntField(env, examples, mappedDataBufferCapacityField);
	int   labelsCapacity = (*env)->GetIntField(env, labels,   mappedDataBufferCapacityField);

	/* Get free pointers (int free [2]) */
	jsize  argc = (*env)->GetArrayLength(env, freeP);
	jlong *argv = (*env)->GetLongArrayElements(env, freeP, 0);
	invalidArgumentException (argc == 2);

	crossbowExecutionContextSchedule (
			env,
			theGPU,
			dataflowId,
			taskId,
			examplesP, examplesCapacity, (int) examplesStartP, (int) examplesEndP,
			labelsP,     labelsCapacity, (int)   labelsStartP, (int)   labelsEndP,
			argv,
			phase,
			bound);

	(*env)->ReleaseLongArrayElements (env, freeP, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_scheduleNext
	(JNIEnv *env, jobject obj,
	jint dataflowId,
	jint taskId,
	jlong examplesStartP, jlong examplesEndP,
	jlong   labelsStartP, jlong   labelsEndP,
	jlongArray freeP,
	jint phase,
	jint bound) {

	(void) obj;

	/* Get free pointers (int free [2]) */
	jsize  argc = (*env)->GetArrayLength(env, freeP);
	jlong *argv = (*env)->GetLongArrayElements(env, freeP, 0);
	invalidArgumentException (argc == 2);

	crossbowExecutionContextScheduleNext (
			env,
			theGPU,
			dataflowId,
			taskId,
			(int) examplesStartP, (int) examplesEndP,
			(int)   labelsStartP, (int)   labelsEndP,
			argv,
			phase,
			bound);

	(*env)->ReleaseLongArrayElements (env, freeP, argv, JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_lockAny
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	return crossbowExecutionContextLockModels (theGPU);
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_merge
	(JNIEnv *env, jobject obj, jboolean pull) {

	(void) env;
	(void) obj;

	int result = crossbowExecutionContextMergeModels (theGPU, (pull == JNI_TRUE) ? 1 : 0);

	return result;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_synchronise
	(JNIEnv *env, jobject obj, jint first, jint clock, jint autotune, jboolean push) {

	(void) env;
	(void) obj;

	return crossbowExecutionContextSynchroniseModels (theGPU, first, clock, autotune, (push == JNI_TRUE) ? 1 : 0);
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_unlockAny
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	return crossbowExecutionContextUnlockModels(theGPU);
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_checkpointModel
	(JNIEnv *env, jobject obj, jstring dir) {
	
	(void) obj;
	
	const char *binding = (*env)->GetStringUTFChars (env, dir, NULL);
	
	crossbowExecutionContextCheckpointModels (theGPU, binding);
	
	(*env)->ReleaseStringUTFChars (env, dir, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_overrideModelData
	(JNIEnv *env, jobject obj, jstring dir) {
	(void) obj;
	const char *binding;
	if (! (*env)->IsSameObject(env, dir, NULL)) {	
		info("Overriding model data...\n");
		binding = (*env)->GetStringUTFChars (env, dir, NULL);
		crossbowExecutionContextOverrideModelData (theGPU, binding);
		(*env)->ReleaseStringUTFChars (env, dir, binding);
	} 
	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_addModel
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	crossbowExecutionContextAddModel (theGPU);

	return 0;
}


JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_delModel
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	crossbowExecutionContextDelModel (theGPU);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetInit
	(JNIEnv *env, jobject obj, jint phase, jint workers, jintArray _capacity, jint NB, jint b, jintArray _padding) {

	(void) obj;

	jint *capacity = (*env)->GetIntArrayElements(env, _capacity, 0);
	jint *padding  = (*env)->GetIntArrayElements(env, _padding,  0);

	crossbowExecutionContextRecordDatasetInit (theGPU, phase, workers, capacity, NB, b, padding);

	(*env)->ReleaseIntArrayElements (env, _capacity, capacity, JNI_ABORT);
	(*env)->ReleaseIntArrayElements (env, _padding,  padding,  JNI_ABORT);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetRegister
	(JNIEnv *env, jobject obj, jint phase, jint id, jstring filename) {

	(void) obj;

	const char *binding = (*env)->GetStringUTFChars (env, filename, NULL);

	crossbowExecutionContextRecordDatasetRegister (theGPU, phase, id, binding);

	(*env)->ReleaseStringUTFChars (env, filename, binding);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetFinalise
	(JNIEnv *env, jobject obj, jint phase) {

	(void) env;
	(void) obj;

	crossbowExecutionContextRecordDatasetFinalise (theGPU, phase);

	return 0;
}
