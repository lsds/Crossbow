#include "uk_ac_imperial_lsds_crossbow_device_blas_BLAS.h"
#include "BLAS.h"

#include <jni.h>

#include <stdlib.h>
#include <stdint.h> /* intptr_t */

#include "bufferpool.h"
#include "bytebuffer.h"

#include "debug.h"

static crossbowBufferPoolP pool;

static jclass class;
static jmethodID writeMethod, readMethod;

static jclass    mappedDataByteBufferClassRef  = NULL;
static jfieldID  mappedDataBufferAddressField  = NULL;
static jfieldID  mappedDataBufferCapacityField = NULL;

static jclass    dataBufferClassRef = NULL;
static jfieldID  dataBufferByteBufferField = NULL;

void writeInput (JNIEnv *, jobject, int, crossbowByteBufferP);
void readOutput (JNIEnv *, jobject, int, crossbowByteBufferP);

void blas_init (int size, int capacity) {
	pool = crossbowBufferPoolCreate (size, capacity);
	/* Setup OpenBLAS threading */
	openblas_set_num_threads(1);
}

void blas_free () {
	crossbowBufferPoolFree (pool);
	return;
}

void *getObjectBufferAddress (JNIEnv *env, jobject obj, int offset) {
	jobject buffer;
	nullPointerException(obj);
	if ((*env)->IsInstanceOf(env, obj, mappedDataByteBufferClassRef)) {
		return (void *) ((intptr_t) (*env)->GetLongField (env, obj, mappedDataBufferAddressField) + offset);
	}
	else {
		/* Assume that object is an instance of a DataBuffer,
		 * backed by a direct byte buffer */
		buffer = (*env)->GetObjectField (env, obj, dataBufferByteBufferField);
		return (void *) ((*env)->GetDirectBufferAddress(env, buffer) + offset);
	}
}

enum CBLAS_TRANSPOSE getCblasTrans (const char *s) {
	// printf("C: transpose string is %s\n", s);
	switch (s[0]) {
		case 'N': return CblasNoTrans;
		case 'n': return CblasNoTrans;
		case 'T': return CblasTrans;
		case 't': return CblasTrans;
		default:  return -1;
	}
	return -1;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_init
(JNIEnv *env, jobject obj, jint size, jint bufferSize) {

	/* Configure callbacks to BLAS */
	jclass localClassRef = (*env)->GetObjectClass (env, obj);
	class = (jclass) (*env)->NewGlobalRef(env, localClassRef);

	writeMethod = (*env)->GetMethodID (env, class, "inputDataMovementCallback",  "(IJI)V");
	nullPointerException (writeMethod);

	readMethod = (*env)->GetMethodID (env, class, "outputDataMovementCallback",  "(IJI)V");
	nullPointerException (readMethod);

	/* Static references to Crossbow's MappedDataBuffer class */
	mappedDataByteBufferClassRef  = (jclass) (*env)->NewGlobalRef
		(env, (*env)->FindClass (env, "uk/ac/imperial/lsds/crossbow/data/MappedDataBuffer"));

	mappedDataBufferAddressField  = (*env)->GetFieldID(env, mappedDataByteBufferClassRef, "address",  "J");
	mappedDataBufferCapacityField = (*env)->GetFieldID(env, mappedDataByteBufferClassRef, "capacity", "I");

	dataBufferClassRef = (jclass) (*env)->NewGlobalRef
		(env, (*env)->FindClass (env, "uk/ac/imperial/lsds/crossbow/data/DataBuffer"));

	dataBufferByteBufferField = (*env)->GetFieldID(env, dataBufferClassRef, "buffer", "Ljava/nio/ByteBuffer;");

	blas_init (size, bufferSize);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_destroy
	(JNIEnv *env, jobject obj) {

	(void) obj;

	(*env)->DeleteGlobalRef(env, class);
	blas_free ();

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_csgemm__Ljava_lang_String_2Ljava_lang_String_2IIIFIIIIFII
	(JNIEnv *env, jobject obj,
	jstring TransA,
	jstring TransB,
	jint M,
	jint N,
	jint K,
	jfloat alpha,
	jint a,
	jint lda,
	jint b,
	jint ldb,
	jfloat beta,
	jint c,
	jint ldc) {

	(void) env;
	(void) obj;

	const char *_TransA = (*env)->GetStringUTFChars(env, TransA, NULL);
	const char *_TransB = (*env)->GetStringUTFChars(env, TransB, NULL);

	enum CBLAS_TRANSPOSE transA = getCblasTrans (_TransA);
	enum CBLAS_TRANSPOSE transB = getCblasTrans (_TransB);

	/* Get buffers */
	crossbowByteBufferP _A = crossbowBufferPoolGet (pool, a);
	crossbowByteBufferP _B = crossbowBufferPoolGet (pool, b);
	crossbowByteBufferP _C = crossbowBufferPoolGet (pool, c);

	nullPointerException (_A);
	nullPointerException (_B);
	nullPointerException (_C);

	/* Copy input buffers */
	writeInput (env, obj, a, _A);
	writeInput (env, obj, b, _B);

	if (beta != 0.)
		writeInput (env, obj, c, _C);

	float *A = (float *) crossbowByteBufferData (_A);
	float *B = (float *) crossbowByteBufferData (_B);
	float *C = (float *) crossbowByteBufferData (_C);

	cblas_sgemm (CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

	/* Copy output buffer(s) */
	readOutput (env, obj, c, _C);

	// Release buffers
	crossbowBufferPoolRelease (pool, a, _A);
	crossbowBufferPoolRelease (pool, b, _B);
	crossbowBufferPoolRelease (pool, c, _C);

	(*env)->ReleaseStringUTFChars (env, TransA, _TransA);
	(*env)->ReleaseStringUTFChars (env, TransB, _TransB);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_csgemm__Ljava_lang_String_2Ljava_lang_String_2IIIFLuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2IIILuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2IIIFLuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2III
	(JNIEnv *env, jobject obj,
	jstring TransA,
	jstring TransB,
	jint M,
	jint N,
	jint K,
	jfloat alpha,
	jobject a,
	jint startA,
	jint endA,
	jint lda,
	jobject b,
	jint startB,
	jint endB,
	jint ldb,
	jfloat beta,
	jobject c,
	jint startC,
	jint endC,
	jint ldc) {

	(void) env;
	(void) obj;

	(void) endA;
	(void) endB;
	(void) endC;

	const char *_TransA = (*env)->GetStringUTFChars(env, TransA, NULL);
	const char *_TransB = (*env)->GetStringUTFChars(env, TransB, NULL);

	enum CBLAS_TRANSPOSE transA = getCblasTrans (_TransA);
	enum CBLAS_TRANSPOSE transB = getCblasTrans (_TransB);

	void *A = getObjectBufferAddress (env, a, startA);
	void *B = getObjectBufferAddress (env, b, startB);
	void *C = getObjectBufferAddress (env, c, startC);

	cblas_sgemm (CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

	(*env)->ReleaseStringUTFChars (env, TransA, _TransA);
	(*env)->ReleaseStringUTFChars (env, TransB, _TransB);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_csgemv__Ljava_lang_String_2IIFIIIIFII
	(JNIEnv *env, jobject obj,
	jstring TransA,
	jint M,
	jint N,
	jfloat alpha,
	jint a,
	jint lda,
	jint x,
	jint incX,
	jfloat beta,
	jint y,
	jint incY) {

	(void) env;
	(void) obj;

	const char *_TransA = (*env)->GetStringUTFChars(env, TransA, NULL);
	enum CBLAS_TRANSPOSE transA = getCblasTrans (_TransA);

	// Get buffers
	crossbowByteBufferP _A = crossbowBufferPoolGet (pool, a);
	crossbowByteBufferP _X = crossbowBufferPoolGet (pool, x);
	crossbowByteBufferP _Y = crossbowBufferPoolGet (pool, y);

	nullPointerException (_A);
	nullPointerException (_X);
	nullPointerException (_Y);

	writeInput (env, obj, a, _A);
	writeInput (env, obj, x, _X);
	if (beta != 0.) {
		writeInput (env, obj, y, _Y);
	}

	float *A = (float *) crossbowByteBufferData (_A);
	float *X = (float *) crossbowByteBufferData (_X);
	float *Y = (float *) crossbowByteBufferData (_Y);

	cblas_sgemv (CblasRowMajor, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);

	readOutput (env, obj, y, _Y);

	crossbowBufferPoolRelease (pool, a, _A);
	crossbowBufferPoolRelease (pool, x, _X);
	crossbowBufferPoolRelease (pool, y, _Y);

	(*env)->ReleaseStringUTFChars (env, TransA, _TransA);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_csgemv__Ljava_lang_String_2IIFLuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2IIILuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2IFLuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2I
	(JNIEnv *env, jobject obj,
	jstring TransA,
	jint M,
	jint N,
	jfloat alpha,
	jobject a,
	jint start,
	jint end,
	jint lda,
	jobject x,
	jint incX,
	jfloat beta,
	jobject y,
	jint incY) {

	(void) env;
	(void) obj;

	(void) end;

	const char *_TransA = (*env)->GetStringUTFChars(env, TransA, NULL);
	enum CBLAS_TRANSPOSE transA = getCblasTrans (_TransA);

	void *A = getObjectBufferAddress (env, a, start);
	void *X = getObjectBufferAddress (env, x, 0);
	void *Y = getObjectBufferAddress (env, y, 0);

	cblas_sgemv (CblasRowMajor, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);

	(*env)->ReleaseStringUTFChars (env, TransA, _TransA);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_csaxpby__IFIIFII
	(JNIEnv *env, jobject obj,
	jint N,
	jfloat alpha,
	jint x,
	jint incX,
	jfloat beta,
	jint y,
	jint incY) {

	(void) env;
	(void) obj;

	crossbowByteBufferP _X = crossbowBufferPoolGet (pool, x);
	crossbowByteBufferP _Y = crossbowBufferPoolGet (pool, y);

	nullPointerException (_X);
	nullPointerException (_Y);

	writeInput (env, obj, x, _X);
	if (beta != 0)
		writeInput (env, obj, y, _Y);

	float *X = (float *) crossbowByteBufferData (_X);
	float *Y = (float *) crossbowByteBufferData (_Y);

	cblas_saxpby (N, alpha, X, incX, beta, Y, incY);

	readOutput (env, obj, y, _Y);

	crossbowBufferPoolRelease (pool, x, _X);
	crossbowBufferPoolRelease (pool, y, _Y);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_blas_BLAS_csaxpby__IFLuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2IIIFLuk_ac_imperial_lsds_crossbow_data_IDataBuffer_2I
	(JNIEnv *env, jobject obj,
	jint N,
	jfloat alpha,
	jobject x,
	jint start,
	jint end,
	jint incX,
	jfloat beta,
	jobject y,
	jint incY) {

	(void) env;
	(void) obj;

	(void) end;

	void *X = getObjectBufferAddress (env, x, start);
	void *Y = getObjectBufferAddress (env, y, 0);

	cblas_saxpby (N, alpha, X, incX, beta, Y, incY);

	return 0;
}

void writeInput (JNIEnv *env, jobject obj, int ndx, crossbowByteBufferP p) {

	void *data =  crossbowByteBufferData (p);
	long address = (long) data;
	int size = crossbowByteBufferSize (p);

	(*env)->CallVoidMethod (env, obj, writeMethod, ndx, address, size);

	return ;
}

void readOutput (JNIEnv *env, jobject obj, int ndx, crossbowByteBufferP p) {

	long address = (long) crossbowByteBufferData (p);
	int size = crossbowByteBufferSize (p);

	(*env)->CallVoidMethod (env, obj, readMethod, ndx, address, size);

	return;
}
