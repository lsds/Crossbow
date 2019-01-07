#include "measurementlist.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowMeasurementListP crossbowMeasurementListCreate (int size, unsigned running) {
	int i;
	crossbowMeasurementListP p;
	p = (crossbowMeasurementListP) crossbowMalloc (sizeof(crossbow_measurementlist_t));
	p->size = (size > 0) ? size : 1;
	p->elements = 0;
	p->measurements = crossbowMalloc (p->size * sizeof(float));
	for (i = 0; i < p->size; ++i)
		p->measurements[i] = 0;
	p->running = running;
	p->avg = 0;
	return p;
}

int crossbowMeasurementListSize (crossbowMeasurementListP p) {
	return p->size;
}

void crossbowMeasurementListResize (crossbowMeasurementListP p, int newsize) {
	int i;
	float *other = crossbowMalloc (newsize * sizeof(float));
	for (i = 0; i < newsize; ++i)
		if (i < p->size)
			other[i] = p->measurements[i];
		else
			other[i] = 0;
	float *t = p->measurements;
	p->measurements = other;
	crossbowFree (t, p->size * sizeof(float));
	p->size = newsize;
	return;
}

int crossbowMeasurementListElements (crossbowMeasurementListP p) {
	return p->elements;
}

int crossbowMeasurementListIsFull (crossbowMeasurementListP p) {
	return (p->elements == p->size);
}

float crossbowMeasurementListGet (crossbowMeasurementListP p, int ndx) {
	nullPointerException(p);
	indexOutOfBoundsException (ndx, p->size);
	return p->measurements[ndx];
}

void crossbowMeasurementListAppend (crossbowMeasurementListP p, float value) {
	int slot;
	if (crossbowMeasurementListIsFull(p))
		crossbowMeasurementListResize(p, 2 * p->size); /* Double the size */
	slot = p->elements ++;
	p->measurements[slot] = value;
	if (p->running) /* Compute running average */
		p->avg += (value - p->avg) / ((float) p->elements);
	return;
}

float crossbowMeasurementListRunningAverage (crossbowMeasurementListP p) {
	return p->avg;
}

float crossbowMeasurementListAverage (crossbowMeasurementListP p) {
	int i;
	float sum = 0;
	if (p->elements == 0)
		return sum;
	for (i = 0; i < p->elements; ++i)
		sum += p->measurements[i];
	return (sum / p->elements);
}

void crossbowMeasurementListFree (crossbowMeasurementListP p) {
	if (! p)
		return;
	crossbowFree (p->measurements, p->size * sizeof(float));
	crossbowFree (p, sizeof(crossbow_measurementlist_t));
	return;
}
