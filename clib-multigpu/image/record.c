#include "record.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "boundingbox.h"

crossbowRecordP crossbowRecordCreate () {
    crossbowRecordP p = NULL;
    p = (crossbowRecordP) crossbowMalloc (sizeof(crossbow_record_t));
    memset (p, 0, sizeof(crossbow_record_t));
    return p;
}

void crossbowRecordReadFromMemory (crossbowRecordP p, void *data, int position) {
	nullPointerException (p);
	(void) data;
    (void) position;
}

void crossbowRecordReadFromFile (crossbowRecordP p, FILE *file, int position) {
	
    int nr;  /* Return value of fread */
    int ndx; /* Generic iterator */
    int N;   /* Number of bounding boxes */
    
	nullPointerException (p);
    
	/* A reminder of how a record is laid out of disk:
     *
	 * Record length
	 * Label
	 * Number of bounding boxes
	 * x-min
	 * y-min
	 * x-max
	 * y-max
	 * Image height
	 * Image width
	 * JPEG
	 */
    
	nr = fread(&(p->length), 4, 1, file); invalidConditionException(nr == 1);
	nr = fread(&(p->label),  4, 1, file); invalidConditionException(nr == 1);
    /* Read bounding boxes, if any */
	nr = fread(&N, 4, 1, file); invalidConditionException(nr == 1);
    if (N > 0) {
        p->boxes = crossbowArrayListCreate (N);
        for (ndx = 0; ndx < N; ++ndx) {
            crossbowBoundingBoxP box = crossbowBoundingBoxCreate ();
            nr = fread(&(box->xmin), 4, 1, file); invalidConditionException(nr == 1);
            nr = fread(&(box->ymin), 4, 1, file); invalidConditionException(nr == 1);
            nr = fread(&(box->xmax), 4, 1, file); invalidConditionException(nr == 1);
            nr = fread(&(box->ymax), 4, 1, file); invalidConditionException(nr == 1);
            /* Store box in array list */
            crossbowArrayListSet (p->boxes, ndx, box);
        }
    }
    /* Read image dimensions */
    nr = fread(&(p->height), 4, 1, file); invalidConditionException(nr == 1);
    nr = fread(&(p->width),  4, 1, file); invalidConditionException(nr == 1);
    /* Read JPEG image */
    p->image = crossbowImageCreate (3, 224, 224);
    crossbowImageReadFromFile (p->image, file);
    crossbowImageStartDecoding (p->image);
    crossbowImageDecode (p->image);
    crossbowImageCast (p->image);
    
	fseek(file, position,  SEEK_SET); /* Reset file pointer to the beginning of this record */
	fseek(file, p->length, SEEK_CUR); /* Increment pointer by record length */
	return;
}

char *crossbowRecordString (crossbowRecordP p) {
	nullPointerException(p);
	char s [2048];
	int offset;
	size_t remaining;
    char *img;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
    img = crossbowImageString (p->image);
	crossbowStringAppend (s, &offset, &remaining, "%7d bytes; label %3d; %d boxes; %4d x %4d pixels; %s}", 
        p->length, 
        p->label, 
        ((p->boxes) ? crossbowArrayListSize(p->boxes) : 0), 
        p->height, p->width,
        img != NULL ? img : "null");
	if (img)
    	crossbowStringFree (img);
	return crossbowStringCopy (s);
}

int crossbowRecordLabelCopy (crossbowRecordP p, void *buffer, int offset, int limit) {

    nullPointerException (p);
    int length = sizeof(int);
    if (limit > 0)
        invalidConditionException ((offset + length) < (limit + 1));
    /* Copy p->label to buffer (starting at offset) */
    memcpy ((void *)(buffer + offset), (void *)(&p->label), length);
    return length;
}

void crossbowRecordFree (crossbowRecordP p) {
    int ndx;
    crossbowBoundingBoxP box;
    if (! p)
        return;
    if (p->boxes) {
        for (ndx = 0; ndx < crossbowArrayListSize(p->boxes); ++ndx) {
            box = (crossbowBoundingBoxP) crossbowArrayListGet (p->boxes, ndx);
            crossbowBoundingBoxFree (box);
        }
        crossbowArrayListFree (p->boxes);
    }
    /* Free image */
    crossbowImageFree (p->image);
    crossbowFree (p, sizeof(crossbow_record_t));
    return;
}
