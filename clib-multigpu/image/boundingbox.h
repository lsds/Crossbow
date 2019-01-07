#ifndef __CROSSBOW_BBOX_H_
#define __CROSSBOW_BBOX_H_

typedef struct crossbow_bbox *crossbowBoundingBoxP;
typedef struct crossbow_bbox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
} crossbow_bbox_t;

crossbowBoundingBoxP crossbowBoundingBoxCreate ();

void crossbowBoundingBoxFree (crossbowBoundingBoxP);

#endif /* __CROSSBOW_BBOX_H_ */
