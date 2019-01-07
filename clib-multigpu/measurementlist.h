#ifndef __CROSSBOW_MEASUREMENTLIST_H_
#define __CROSSBOW_MEASUREMENTLIST_H_

typedef struct crossbow_measurementlist *crossbowMeasurementListP;
typedef struct crossbow_measurementlist {
	int size;
	int elements;
	float *measurements;
	unsigned running;
	float avg;
} crossbow_measurementlist_t;

crossbowMeasurementListP crossbowMeasurementListCreate (int size, unsigned running);

int crossbowMeasurementListSize (crossbowMeasurementListP);

void crossbowMeasurementListResize (crossbowMeasurementListP, int);

int crossbowMeasurementListElements (crossbowMeasurementListP);

int crossbowMeasurementListIsFull (crossbowMeasurementListP);

float crossbowMeasurementListGet (crossbowMeasurementListP, int);

void crossbowMeasurementListAppend (crossbowMeasurementListP, float);

float crossbowMeasurementListRunningAverage (crossbowMeasurementListP);

float crossbowMeasurementListAverage (crossbowMeasurementListP);

void crossbowMeasurementListFree (crossbowMeasurementListP);

#endif /* __CROSSBOW_MEASUREMENTLIST_H_ */
