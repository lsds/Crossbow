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

void crossbowBoundingBoxFree (crossbowBoundingBoxP p) {
	if (! p)
		return;
	crossbowFree (p, sizeof(crossbow_bbox_t));
}
