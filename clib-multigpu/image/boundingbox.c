#include "boundingbox.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

crossbowBoundingBoxP crossbowBoundingBoxCreate () {
	crossbowBoundingBoxP p = NULL;
	p = (crossbowBoundingBoxP) crossbowMalloc (sizeof(crossbow_bbox_t));
	p->xmin = -1;
	p->ymin = -1;
	p->xmax = -1;
	p->ymax = -1;
	return p;
}

unsigned crossbowBoundingBoxIsValid (crossbowBoundingBoxP p) {
	nullPointerException(p);
	/* All values must be in [0,1] */
	if (
	((p->xmin < 0.0) || (p->xmin > 1.0)) ||
	((p->ymin < 0.0) || (p->ymin > 1.0)) ||
	((p->xmax < 0.0) || (p->xmax > 1.0)) ||
	((p->ymax < 0.0) || (p->ymax > 1.0))
	) {
		return 0;
	}
	return 1;
}

void crossbowBoundingBoxFree (crossbowBoundingBoxP p) {
	if (! p)
		return;
	crossbowFree (p, sizeof(crossbow_bbox_t));
}

void crossbowBoundingBoxDump (crossbowBoundingBoxP p) {
	nullPointerException (p);
	printf("%.5f %.5f; %.5f %.5f\n", p->xmin, p->ymin, p->xmax, p->ymax);
}

