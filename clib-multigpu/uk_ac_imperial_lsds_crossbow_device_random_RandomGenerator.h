/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator */

#ifndef _Included_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
#define _Included_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
 * Method:    test
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_test
  (JNIEnv *, jobject);

/*
 * Class:     uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
 * Method:    init
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_init
  (JNIEnv *, jobject, jlong);

/*
 * Class:     uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
 * Method:    randomUniformFill
 * Signature: (Ljava/nio/ByteBuffer;IFF)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_randomUniformFill
  (JNIEnv *, jobject, jobject, jint, jfloat, jfloat);

/*
 * Class:     uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
 * Method:    randomGaussianFill
 * Signature: (Ljava/nio/ByteBuffer;IFFI)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_randomGaussianFill
  (JNIEnv *, jobject, jobject, jint, jfloat, jfloat, jint);

/*
 * Class:     uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator
 * Method:    destroy
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_random_RandomGenerator_destroy
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
