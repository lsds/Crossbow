#ifndef __CROSSBOW_MEMORYMANAGER_H_
#define __CROSSBOW_MEMORYMANAGER_H_

#include <stddef.h>
#include <stdarg.h>

#include  <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h> /* uintptr_t */
#include <unistd.h> /* getpagesize() */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "utils.h"

void *crossbowMalloc (int);

void *crossbowMallocAligned (int, int);

void *crossbowCudaMallocHost (int);

void *crossbowCudaMalloc (int);

void *crossbowFree (void *, int);

void *crossbowCudaFreeHost (void *, int);

void *crossbowCudaFree (void *, int);

char *crossbowStringCopy (const char *);

int crossbowStringAppend (char *, int *, size_t *, const char *format, ...);

char *crossbowStringConcat (const char *format, ...);

void *crossbowStringFree (const char *);

void crossbowMemoryManagerInit ();

void crossbowMemoryManagerDestroy ();

void crossbowHostRegisterBuffer (int, void *, int, int, int, crossbowPhase_t);

void crossbowMemoryManagerDump ();

#endif /* __CROSSBOW_MEMORYMANAGER_H_ */
