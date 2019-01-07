#include "rectangle.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

crossbowRectangleP crossbowRectangleCreate (int xmin, int ymin, int xmax, int ymax) {
    crossbowRectangleP p = NULL;
    p = (crossbowRectangleP) crossbowMalloc (sizeof(crossbow_rectangle_t));
    p->xmin = xmin;
    p->ymin = ymin;
    p->xmax = xmax;
    p->ymax = ymax;
    return p;
}

void crossbowRectangleSet (crossbowRectangleP p, int xmin, int ymin, int xmax, int ymax) {
    nullPointerException (p);
    p->xmin = xmin;
    p->ymin = ymin;
    p->xmax = xmax;
    p->ymax = ymax;
    return;
}

float crossbowRectangleArea (crossbowRectangleP p) {
    float x = (float) (p->xmax - p->xmin);
    float y = (float) (p->ymax - p->ymin);
    return (x * y);
}

unsigned crossbowRectangleEmpty (crossbowRectangleP p) {
    nullPointerException (p);
    return ((p->xmin > p->xmax) || (p->ymin > p->ymax));
}

unsigned crossbowRectangleValid (crossbowRectangleP p, float limit) {
    nullPointerException (p);
    float area = crossbowRectangleArea (p);
    return (area >= limit);
}

crossbowRectangleP crossbowRectangleIntersect (crossbowRectangleP p, crossbowRectangleP q) {
    nullPointerException (p);
    nullPointerException (q);
    int xmin = max(p->xmin, q->xmin);
    int ymin = max(p->ymin, q->ymin);
    int xmax = max(p->xmax, q->xmax);
    int ymax = max(p->ymax, q->ymax);
    if ((xmin > xmax) || (ymin > ymax))
        return crossbowRectangleCreate (0, 0, 0, 0);
    else
        return crossbowRectangleCreate (xmin, ymin, xmax, ymax);
}

/*
 * Determine if rectangle `p` covers a sufficient
 * fraction of the bounding boxes.
 */
unsigned crossbowRectangleCovers (crossbowRectangleP p, float limit, crossbowArrayListP boxes) {

	nullPointerException (p);

	if (! crossbowRectangleValid (p, 1))
		return 0;

	unsigned covered = 0;
	crossbowRectangleP intersection;
	int i;
	float coverage;

	int length = crossbowArrayListSize (boxes);
	crossbowRectangleP box;

	/* Iterate over `length` bounding boxes */
	for (i = 0; i < length; ++i) {
		box = (crossbowRectangleP) crossbowArrayListGet (boxes, i);
		if (! crossbowRectangleValid (box, 1))
			continue;
		intersection = crossbowRectangleIntersect (p, box);
		coverage = crossbowRectangleArea (intersection) / crossbowRectangleArea (p);
		crossbowFree (intersection, sizeof(crossbow_rectangle_t));
		if (coverage >= limit) {
			covered = 1;
			break;
		}
	}

	return covered;
}

void crossbowRectangleFree (crossbowRectangleP p) {
    if (! p)
        return;
    crossbowFree (p, sizeof(crossbow_rectangle_t));
}
