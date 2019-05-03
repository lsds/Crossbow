#ifndef __CROSSBOW_RECTANGLE_H_
#define __CROSSBOW_RECTANGLE_H_

#include "../arraylist.h"

typedef struct crossbow_rectangle *crossbowRectangleP;
typedef struct crossbow_rectangle {
    int xmin;
    int ymin;
    int xmax;
    int ymax;
} crossbow_rectangle_t;

crossbowRectangleP crossbowRectangleCreate (int, int, int, int);

void crossbowRectangleSet (crossbowRectangleP, int, int, int, int);

float crossbowRectangleArea (crossbowRectangleP);

unsigned crossbowRectangleEmpty (crossbowRectangleP);

unsigned crossbowRectangleValid (crossbowRectangleP, float);

crossbowRectangleP crossbowRectangleIntersect (crossbowRectangleP, crossbowRectangleP);

unsigned crossbowRectangleCovers (crossbowRectangleP, float, crossbowArrayListP);

void crossbowRectangleFree (crossbowRectangleP);

#endif /* __CROSSBOW_RECTANGLE_H_ */
