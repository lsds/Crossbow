#ifndef __CROSSBOW_RECORDREADER_H_
#define __CROSSBOW_RECORDREADER_H_

#include "../list.h"

#include "record.h"
#include "recordfile.h"

typedef struct crossbow_record_reader *crossbowRecordReaderP;
typedef struct crossbow_record_reader {
    crossbowListP dataset;
    int records; /* Total number of records in dataset */
    int counter; /* Current record counter */
    int limit;
    int wraps;
    crossbowRecordFileP current;
    unsigned finalised;
    int workers;
    int jc; /* Pin workers to cores, starting from core `jc` */
} crossbow_record_reader_t;

typedef struct crossbow_record_reader_task *crossbowRecordReaderTaskP;
typedef struct crossbow_record_reader_task {
    /* Worker id */
    int id;
    int jc;
    int counter;
    /* Read from file at position */
    crossbowRecordFileP file;
    int position;
    /* Write to buffer at offset */
    void *buffer[2];
    int   offset[2];
} crossbow_record_reader_task_t;

crossbowRecordReaderP crossbowRecordReaderCreate (int);

void crossbowRecordReaderCoreOffset (crossbowRecordReaderP, int);

void crossbowRecordReaderRegister (crossbowRecordReaderP, const char *);

void crossbowRecordReaderFinalise (crossbowRecordReaderP);

void crossbowRecordReaderRepeat (crossbowRecordReaderP, int);

unsigned crossbowRecordReaderHasNext (crossbowRecordReaderP);

void crossbowRecordReaderNext (crossbowRecordReaderP, crossbowRecordP);

crossbowRecordFileP crossbowRecordReaderNextPointer (crossbowRecordReaderP, int *);

void crossbowRecordReaderRead (crossbowRecordReaderP, int, int, void *, int);

void crossbowRecordReaderReadProperly (crossbowRecordReaderP, int, int *, int, int *, void *, void *, int *);

void crossbowRecordReaderFree (crossbowRecordReaderP);

#endif /* __CROSSBOW_RECORDREADER_H_ */
