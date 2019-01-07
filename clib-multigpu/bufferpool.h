#ifndef __CROSSBOW_BUFFERPOOL_H_
#define __CROSSBOW_BUFFERPOOL_H_

#include "bytebuffer.h"

typedef struct crossbow_bufferpool *crossbowBufferPoolP;
typedef struct crossbow_bufferpool {
	int size;
	int default_capacity;
	crossbowByteBufferP *buffers;
} crossbow_bufferpool_t;

crossbowBufferPoolP crossbowBufferPoolCreate (int, int);

crossbowByteBufferP crossbowBufferPoolGet (crossbowBufferPoolP, int);

void crossbowBufferPoolRelease (crossbowBufferPoolP, int, crossbowByteBufferP);

int crossbowBufferPoolSize (crossbowBufferPoolP);

void crossbowBufferPoolFree (crossbowBufferPoolP);

#endif /* __CROSSBOW_BUFFERPOOL_H_ */
