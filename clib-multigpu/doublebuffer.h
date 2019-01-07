#ifndef __CROSSBOW_DOUBLEBUFFER_H_
#define __CROSSBOW_DOUBLEBUFFER_H_

#include <pthread.h>

/* 
 * A page-aligned, page-locked memory region to store 
 * pre-processed JPEG images and their labels.
 */
typedef struct crossbow_doublebuffer *crossbowDoubleBufferP;
typedef struct crossbow_doublebuffer {
    
	int NB; /* Number of batches to store in buffer(s) */
	
    /* Batch size and padding (for image and label batch) */
    int b;
	int padding[2];

	int size[2];
    
    /* Double buffer size (in bytes) */
    int capacity[2];
    
	void *theImages[2];
	void *theLabels[2];
    
	int idx; /* Pointer to one of the two buffers (same for images and labels) */
    
    volatile unsigned needed;
    volatile unsigned locked;

    /* Locking mechanism */
    void *slots;
	
} crossbow_doublebuffer_t;

crossbowDoubleBufferP crossbowDoubleBufferCreate (int *, int, int, int *);

int *crossbowDoubleBufferCapacity (crossbowDoubleBufferP);

void crossbowDoubleBufferRegister (crossbowDoubleBufferP);

void crossbowDoubleBufferAdviceWillNeed (crossbowDoubleBufferP);

void crossbowDoubleBufferUnregister (crossbowDoubleBufferP);

void crossbowDoubleBufferLock (crossbowDoubleBufferP, int);

void crossbowDoubleBufferUnlock (crossbowDoubleBufferP, int);

void crossbowDoubleBufferFree (crossbowDoubleBufferP);

#endif /* __CROSSBOW_DOUBLEBUFFER_H_ */
