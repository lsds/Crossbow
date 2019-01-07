#include "sleep.h"

/* Busy wait for N cycles */
__global__ void crossbowKernelSleepImpl (int64_t N) {
	int64_t cycles = 0;
	int64_t start = clock64();
	while (cycles < N) {
		cycles = clock64() - start;
	}
}

void crossbowKernelSleep (void *args) {

	int64_t cycles = *((int64_t *) args);

	int blockSize = 1;
	int  gridSize = 1;

	/* info ("Sleep for %ld cycles\n", cycles); */
	crossbowKernelSleepImpl<<< gridSize, blockSize >>>(cycles);

	return;
}
