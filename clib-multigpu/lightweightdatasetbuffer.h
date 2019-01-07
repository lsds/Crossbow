#ifndef __CROSSBOW_LIGHTWEIGHTDATASETBUFFER_H_
#define __CROSSBOW_LIGHTWEIGHTDATASETBUFFER_H_


/*
 * A page-aligned, page-locked memory region
 */
typedef struct crossbow_lightweightdatasetbuffer *crossbowLightWeightDatasetBufferP;
typedef struct crossbow_lightweightdatasetbuffer {
    int capacity;
	void *data;
	volatile unsigned locked;
	volatile unsigned needed;
} crossbow_lightweightdatasetbuffer_t;

crossbowLightWeightDatasetBufferP crossbowLightWeightDatasetBufferCreate (int);

unsigned long crossbowLightWeightDatasetBufferAddress (crossbowLightWeightDatasetBufferP);

int crossbowLightWeightDatasetBufferCapacity (crossbowLightWeightDatasetBufferP);

void crossbowLightWeightDatasetBufferRegister (crossbowLightWeightDatasetBufferP, int);

void crossbowLightWeightDatasetBufferAdviceWillNeed (crossbowLightWeightDatasetBufferP);

void crossbowLightWeightDatasetBufferUnregister (crossbowLightWeightDatasetBufferP, int);

void crossbowLightWeightDatasetBufferAdviceDontNeed (crossbowLightWeightDatasetBufferP);

void crossbowLightWeightDatasetBufferCopy (crossbowLightWeightDatasetBufferP, int, void *, int, int);

void crossbowLightWeightDatasetBufferFree (crossbowLightWeightDatasetBufferP, int);

#endif /* __CROSSBOW_LIGHTWEIGHTDATASETBUFFER_H_ */
