#ifndef __CROSSBOW_RECORDFILE_H_
#define __CROSSBOW_RECORDFILE_H_

#include "record.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fcntl.h> /* open */

#include <sys/mman.h> /* mmap, munmap */

#include <string.h>
#include <stdio.h>

#include <errno.h>

typedef struct crossbow_record_file *crossbowRecordFileP;
typedef struct crossbow_record_file {
	char *filename;
    int workers;
#ifdef MAP_RECORDS
	/* Create a file descriptor and map it into memory (`data` pointer) */
	int fd;
	void *data;
	/* Flags indicate that file has been mapped and the OS has been advised */
	unsigned mapped;
	unsigned needed;
	/*
	 * Somehow we need to maintain the
	 * current read position.
	 */
	int offset;
#else
	/* The alternative is to read records in file with `fread` */
	FILE *fp;
    FILE **f;
#endif
	int length;
	/* The total number of records in the file (stored in file header) */
	int records;
	unsigned opened;
	int counter;
} crossbow_record_file_t;

crossbowRecordFileP crossbowRecordFileCreate (const char *, int);

void crossbowRecordFileOpen (crossbowRecordFileP);

void crossbowRecordFileStat (crossbowRecordFileP);

void crossbowRecordFileMap (crossbowRecordFileP);

void crossbowRecordFileAdviceWillNeed (crossbowRecordFileP);

int crossbowRecordFileHeader (crossbowRecordFileP);

int crossbowRecordFilePosition (crossbowRecordFileP);

int crossbowRecordFileRemaining (crossbowRecordFileP);

unsigned crossbowRecordFileHasRemaining (crossbowRecordFileP);

void crossbowRecordFileReset (crossbowRecordFileP, unsigned);

void crossbowRecordFileRead (crossbowRecordFileP, crossbowRecordP);

int crossbowRecordFileNextPointer (crossbowRecordFileP);

void crossbowRecordFileReadSafely (crossbowRecordFileP, int, int, crossbowRecordP);

void crossbowRecordFileAdviceDontNeed (crossbowRecordFileP);

void crossbowRecordFileUnmap (crossbowRecordFileP);

void crossbowRecordFileClose (crossbowRecordFileP);

void crossbowRecordFileFree (crossbowRecordFileP);

#endif /* __CROSSBOW_RECORDFILE_H_ */
