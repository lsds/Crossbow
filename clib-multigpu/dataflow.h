#ifndef __CROSSBOW_DATAFLOW_H_
#define __CROSSBOW_DATAFLOW_H_

#include "operator.h"

typedef struct crossbow_dataflow *crossbowDataflowP;
typedef struct crossbow_dataflow {
	int id;
	int autoincrement;
	crossbowOperatorP head, tail; /* Doubly-linked list of operators (kernel containers) */
	/*
	 * A pointer to the operator that computes loss. Can be NULL.
	 * A dataflow should have at most one such operator.
	 */
	crossbowOperatorP lossOp;
	crossbowOperatorP accuracyOp;
	crossbowOperatorP dataTransformOp;
} crossbow_dataflow_t;

crossbowDataflowP crossbowDataflowCreate (int);

void crossbowDataflowAppend (crossbowDataflowP, crossbowKernelP, int);

crossbowOperatorP crossbowDataflowPeek (crossbowDataflowP);

int crossbowDataflowMostUpstream (crossbowDataflowP, crossbowOperatorP);

int crossbowDataflowMostDownstream (crossbowDataflowP, crossbowOperatorP);

crossbowOperatorP crossbowDataflowFindKernel (crossbowDataflowP, int);

crossbowOperatorP crossbowDataflowFindOperator (crossbowDataflowP, int);

void crossbowDataflowSetLossOperator (crossbowDataflowP, int);

void crossbowDataflowSetAccuracyOperator (crossbowDataflowP, int);

void crossbowDataflowSetDataTransformOperator (crossbowDataflowP, int);

int crossbowDataflowSize (crossbowDataflowP);

void crossbowDataflowFree (crossbowDataflowP);

void crossbowDataflowDump (crossbowDataflowP);

void crossbowDataflowDumpDependencyGraph (crossbowDataflowP);

#endif /* __CROSSBOW_DATAFLOW_H_ */
