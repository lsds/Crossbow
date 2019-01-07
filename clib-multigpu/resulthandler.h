#ifndef __CROSSBOW_RESULTHANDLER_H_
#define __CROSSBOW_RESULTHANDLER_H_

typedef struct crossbow_resulthandler *crossbowResultHandlerP;
typedef struct crossbow_resulthandler {
	void **slots;
	int *count;
	int offset;
} crossbow_resulthandler_t;

crossbowResultHandlerP crossbowResultHandlerCreate (int);

void  crossbowResultHandlerSet (crossbowResultHandlerP, int, void *, int);

void *crossbowResultHandlerGet (crossbowResultHandlerP, int);

int crossbowResultHandlerNumberOfSlots (crossbowResultHandlerP, int);

void crossbowResultHandlerReserveSlot (crossbowResultHandlerP, int, int, long *, float, float);

void crossbowResultHandlerFree (crossbowResultHandlerP);

#endif /* __CROSSBOW_RESULTHANDLER_H_ */
