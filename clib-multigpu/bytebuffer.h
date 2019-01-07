#ifndef __CROSSBOW_BYTEBUFFER_H_
#define __CROSSBOW_BYTEBUFFER_H_

typedef struct crossbow_bytebuffer *crossbowByteBufferP;
typedef struct crossbow_bytebuffer {
	int size;
	void *data;
} crossbow_bytebuffer_t;

crossbowByteBufferP crossbowByteBufferCreate (int);

void crossbowByteBufferFree (crossbowByteBufferP);

void crossbowByteBufferClear (crossbowByteBufferP);

int crossbowByteBufferSize (crossbowByteBufferP);

void *crossbowByteBufferData (crossbowByteBufferP);

void crossbowByteBufferSwap (crossbowByteBufferP, int);

#endif /* __CROSSBOW_BYTEBUFFER_H_ */
