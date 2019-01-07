#ifndef __CROSSBOW_BLAS_H_
#define __CROSSBOW_BLAS_H_

#include <cblas.h>

#include <jni.h>

void blas_init (int size, int bufferSize);

void blas_free ();

enum CBLAS_TRANSPOSE getCblasTrans (const char *);

void *getObjectBufferAddress (JNIEnv *, jobject, int);

#endif /* __CROSSBOW_BLAS_H_ */
