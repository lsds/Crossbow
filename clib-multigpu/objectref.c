#include "uk_ac_imperial_lsds_crossbow_device_ObjectRef.h"

#include <jni.h>

static jclass globalClassRef;
static jmethodID constructor, getIntValue;
static jobject globalObjectRef;

JNIEXPORT void JNICALL Java_uk_ac_imperial_lsds_crossbow_device_ObjectRef_create
	(JNIEnv *env, jobject obj, jint value) {

	(void) obj;

	jclass localClassRef;
	jobject localObjectRef;

	printf("[Test %s]\n", __func__);

	localClassRef  = (*env)->FindClass (env, "java/lang/Integer");
	globalClassRef = (jclass) (*env)->NewGlobalRef (env, localClassRef);

	constructor = (*env)->GetMethodID (env, globalClassRef,   "<init>", "(I)V");
	getIntValue = (*env)->GetMethodID (env, globalClassRef, "intValue",  "()I");

	localObjectRef = (*env)->NewObject (env, globalClassRef, constructor, value);
	globalObjectRef = (*env)->NewGlobalRef (env, localObjectRef);

	return;
}

JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_ObjectRef_get
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	printf("[Return %p]\n", globalObjectRef);

	return globalObjectRef;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_ObjectRef_test
	(JNIEnv *env, jobject obj, jobject var) {

	(void) obj;

	jboolean result;

	printf("[Test %s]\n", __func__);

	int value = (*env)->CallIntMethod (env, var, getIntValue);

	printf("[Compare variable %p (int value %d) with %p]\n", var, value, globalObjectRef);

	result = (*env)->IsSameObject (env, globalObjectRef, var);
	if (result == JNI_TRUE)
		return 1;

	return 0;
}

JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_ObjectRef_testAndGet
	(JNIEnv *env, jobject obj, jobject var) {

	(void) env;
	(void) obj;
	(void) var;

	printf("[Test %s]\n", __func__);

	return NULL;
}
