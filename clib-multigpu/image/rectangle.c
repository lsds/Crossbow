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
	nullPointerException (p);
	/* info("Find area of rectangle from (%d, %d) to (%d, %d)\n", p->xmin, p->ymin, p->xmax, p->ymax); */
	
	/*
	if (p->xmax <= p->xmin) {
		info("Invalid x-axis in rectangle ((%d, %d) (%d, %d))\n",  p->xmin, p->ymin, p->xmax, p->ymax);
	}
	invalidConditionException (p->xmax > p->xmin);
	if (p->ymax <= p->ymin) {
		info("Invalid y-axis in rectangle ((%d, %d) (%d, %d))\n",  p->xmin, p->ymin, p->xmax, p->ymax);
	}
	invalidConditionException (p->ymax > p->ymin);
	*/
	
    float x = (float) (p->xmax - p->xmin);
    float y = (float) (p->ymax - p->ymin);
    return (x * y);
}

unsigned crossbowRectangleEmpty (crossbowRectangleP p) {
    nullPointerException (p);
    return ((p->xmin >= p->xmax) || (p->ymin >= p->ymax));
}

/* Return false is a rectangle covers an area less than `limit` */
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
		/* Return empty rectangle */
        return crossbowRectangleCreate (0, 0, 0, 0);
    else
        return crossbowRectangleCreate (xmin, ymin, xmax, ymax);
}

/*
 * Determine if rectangle `p` covers a sufficient
 * fraction of an array of rectangles.
 */
unsigned crossbowRectangleCovers (crossbowRectangleP p, float limit, crossbowArrayListP boxes) {

	nullPointerException (p);
	
	/* Reject any rectangle which contains no pixels */
	if (! crossbowRectangleValid (p, 1))
		return 0;
	
	unsigned covered = 0;
	crossbowRectangleP intersection;
	int i;
	float coverage;
	
	int length = crossbowArrayListSize (boxes);
	crossbowRectangleP box;

	/* Iterate over all rectangles in array list */
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
