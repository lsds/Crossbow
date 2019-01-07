#ifndef __CROSSBOW_RECORD_H_
#define __CROSSBOW_RECORD_H_

#include "../arraylist.h"
#include "image.h"

#include <stdio.h>

typedef struct crossbow_record *crossbowRecordP;
typedef struct crossbow_record {
    int length;
    int label;
    crossbowArrayListP boxes;
    int height;
    int width;
    crossbowImageP image;
} crossbow_record_t;

crossbowRecordP crossbowRecordCreate ();

void crossbowRecordReadFromMemory (crossbowRecordP, void *, int);

void crossbowRecordReadFromFile (crossbowRecordP, FILE *, int);

char *crossbowRecordString (crossbowRecordP);

int crossbowRecordLabelCopy (crossbowRecordP, void *, int, int);

void crossbowRecordFree (crossbowRecordP);

#endif /* __CROSSBOW_RECORD_H_ */
