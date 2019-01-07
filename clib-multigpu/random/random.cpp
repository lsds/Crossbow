#include "../uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator.h"

#include "generator.hpp"

using namespace crossbow;

static CrossbowRandomGenerator *generator = NULL;

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_test
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	fprintf(stdout, "Test.\n");
	fflush (stdout);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_init
	(JNIEnv *env, jobject obj, jlong seed) {

	(void) env;
	(void) obj;

	generator = new CrossbowRandomGenerator (seed);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_randomUniformFill
(JNIEnv *env, jobject obj, jobject buffer, jint count, jfloat start, jfloat end) {

	(void) obj;

	float *data = (float *) (env->GetDirectBufferAddress(buffer));

	generator->randomUniformFill (data, count, start, end);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_randomGaussianFill
	(JNIEnv *env, jobject obj, jobject buffer, jint count, jfloat mean, jfloat std, jint truncate) {

	(void) obj;

	float *data = (float *) (env->GetDirectBufferAddress(buffer));

	generator->randomGaussianFill (data, count, mean, std, truncate);

	return 0;
}

JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_destroy
	(JNIEnv *env, jobject obj) {

	(void) env;
	(void) obj;

	return 0;
}

// } /* namespace crossbow */
