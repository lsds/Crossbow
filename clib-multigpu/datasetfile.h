#ifndef __CROSSBOW_DATASETFILE_H_
#define __CROSSBOW_DATASETFILE_H_

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fcntl.h> /* open */

#include <sys/mman.h> /* mmap, munmap */

#include <errno.h>

#include <cuda.h>
#include <cuda_runtime.h>

typedef struct crossbow_datasetfile *crossbowDatasetFileP;
typedef struct crossbow_datasetfile {
	char *filename;
	int fd;
	void *data;
	int length;
	unsigned opened;
	unsigned mapped;
	volatile unsigned locked;
	volatile unsigned needed;
	void *region;
} crossbow_datasetfile_t;

crossbowDatasetFileP crossbowDatasetFileCreate (const char *);

void crossbowDatasetFileOpen (crossbowDatasetFileP);

void crossbowDatasetFileStat (crossbowDatasetFileP);

void crossbowDatasetFileMap (crossbowDatasetFileP);

void crossbowDatasetFileAssign (crossbowDatasetFileP, void *);

void crossbowDatasetFileWithdraw (crossbowDatasetFileP, void *);

void crossbowDatasetFileRegister (crossbowDatasetFileP, int);

void crossbowDatasetFileRegisterRegion (crossbowDatasetFileP, int, int);

void crossbowDatasetFileAdviceWillNeed (crossbowDatasetFileP);

void crossbowDatasetFileAdviceWillNeedRegion (crossbowDatasetFileP, int, int);

unsigned long crossbowDatasetFileAddress (crossbowDatasetFileP);

int crossbowDatasetFileSize (crossbowDatasetFileP);

void crossbowDatasetFileAdviceDontNeed (crossbowDatasetFileP);

void crossbowDatasetFileAdviceDontNeedRegion (crossbowDatasetFileP, int, int);

void crossbowDatasetFileUnregister (crossbowDatasetFileP, int);

void crossbowDatasetFileUnregisterRegion (crossbowDatasetFileP, int, int);

void crossbowDatasetFileUnmap (crossbowDatasetFileP);

void crossbowDatasetFileClose (crossbowDatasetFileP);

void crossbowDatasetFileFree (crossbowDatasetFileP, int);

#endif /* __CROSSBOW_DATASETFILE_H_ */
