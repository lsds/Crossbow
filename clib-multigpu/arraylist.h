#ifndef __CROSSBOW_ARRAYLIST_H_
#define __CROSSBOW_ARRAYLIST_H_

typedef struct crossbow_arraylist *crossbowArrayListP;
typedef struct crossbow_arraylist {
	int size;
	/*
	 * Iterator index to return element at position
	 * 0, 1, 2, ..., (size - 1), 0, 1, 2, and so on
	 */
	volatile int iter;
	void **elements;
} crossbow_arraylist_t;

crossbowArrayListP crossbowArrayListCreate (int size);

int crossbowArrayListSize (crossbowArrayListP);

void crossbowArrayListResize (crossbowArrayListP, int);

void *crossbowArrayListGet (crossbowArrayListP, int);

void crossbowArrayListSet (crossbowArrayListP, int, void *);

void crossbowArrayListResetIterator (crossbowArrayListP);

void *crossbowArrayListGetNext (crossbowArrayListP);

void *crossbowArrayListGetNextSafely (crossbowArrayListP);

void crossbowArrayListFree (crossbowArrayListP);

#endif /* __CROSSBOW_ARRAYLIST_H_ */
