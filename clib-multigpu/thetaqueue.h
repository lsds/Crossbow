#ifndef __CROSSBOW_THETA_QUEUE_H_
#define __CROSSBOW_THETA_QUEUE_H_

#include "utils.h"

typedef struct crossbow_theta_queue *crossbowThetaQueueP;
typedef struct crossbow_theta_queue {
	volatile int size;
	volatile int iter; /* Iterator index to return element at position 0, 1, 2, ..., (size - 1), 0, 1, 2, and so on */
	void *slots;
	void **elements;
} crossbow_theta_queue_t;

crossbowThetaQueueP crossbowThetaQueueCreate (int);

int crossbowThetaQueueSize (crossbowThetaQueueP);

void crossbowThetaQueueSet (crossbowThetaQueueP, int, void *);

void *crossbowThetaQueueGet (crossbowThetaQueueP, int);

void *crossbowThetaQueueGetNext (crossbowThetaQueueP);

void *crossbowThetaQueueGetNextSafely (crossbowThetaQueueP);

void crossbowThetaQueueReserve (crossbowThetaQueueP, int);

void crossbowThetaQueueRelease (crossbowThetaQueueP, int);

int crossbowThetaQueueIsEnabled (crossbowThetaQueueP, int);

void crossbowThetaQueueEnable (crossbowThetaQueueP, int);

int crossbowThetaQueueIsDisabled (crossbowThetaQueueP, int);

int crossbowThetaQueueDisable (crossbowThetaQueueP, int);

int crossbowThetaQueueDisableAny (crossbowThetaQueueP);

void crossbowThetaQueueExpand (crossbowThetaQueueP, void *);

int crossbowThetaQueueShrink (crossbowThetaQueueP, void *);

void crossbowThetaQueueFree (crossbowThetaQueueP);

#endif /* __CROSSBOW_THETA_QUEUE_H_ */
