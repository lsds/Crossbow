#include "dataflow.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "operatordependency.h"

crossbowDataflowP crossbowDataflowCreate (int id) {
	crossbowDataflowP p;
	p = (crossbowDataflowP) crossbowMalloc (sizeof(crossbow_dataflow_t));
	p->id = id;
	p->head = crossbowOperatorCreate (NULL, -1);
	p->tail = crossbowOperatorCreate (NULL, -1);
	p->head->next = p->tail;
	p->tail->prev = p->head;
	p->autoincrement = 0;
	/* Loss and accuracy operators are initialised to null */
	p->lossOp = NULL;
	p->accuracyOp = NULL;
	p->dataTransformOp = NULL;
	return p;
}

void crossbowDataflowAppend (crossbowDataflowP dataflow, crossbowKernelP kernel, int events) {
	int id = dataflow->autoincrement++;
	crossbowOperatorP p = crossbowOperatorCreate (kernel, id);
	crossbowOperatorConfigure (p, events);
	crossbowOperatorP last = dataflow->tail->prev;
	p->next = dataflow->tail;
	p->prev = last;
	dataflow->tail->prev = p;
	last->next = p;
	return;
}

crossbowOperatorP crossbowDataflowPeek (crossbowDataflowP p) {
	return (p->head->next);
}

int crossbowDataflowMostUpstream (crossbowDataflowP p, crossbowOperatorP op) {
	return (p->head->next == op) || (op->prev == p->dataTransformOp);
}

int crossbowDataflowMostDownstream (crossbowDataflowP p, crossbowOperatorP op) {
	return (p->tail->prev == op);
}

crossbowOperatorP crossbowDataflowFindKernel (crossbowDataflowP p, int kernelId) {
	crossbowOperatorP curr = p->head->next;
	while (curr != p->tail) {
		if (curr->kernel->id == kernelId) return curr;
		/* Peek next */
		curr = curr->next;
	}
	/* Not found */
	return NULL;
}

crossbowOperatorP crossbowDataflowFindOperator (crossbowDataflowP p, int id) {
	crossbowOperatorP curr = p->head->next;
	while (curr != p->tail) {
		if (curr->id == id) return curr;
		/* Peek next */
		curr = curr->next;
	}
	/* Not found */
	return NULL;
}

void crossbowDataflowSetLossOperator (crossbowDataflowP p, int id) {
	crossbowOperatorP lossOp = crossbowDataflowFindKernel(p, id);
	nullPointerException(lossOp);
	invalidConditionException(crossbowKernelOutputPull(lossOp->kernel));
	invalidConditionException((! p->lossOp));
	p->lossOp = lossOp;
}

void crossbowDataflowSetAccuracyOperator (crossbowDataflowP p, int id) {
	crossbowOperatorP accuracyOp = crossbowDataflowFindKernel(p, id);
	nullPointerException(accuracyOp);
	invalidConditionException(crossbowKernelOutputPull(accuracyOp->kernel));
	invalidConditionException((! p->accuracyOp));
	p->accuracyOp = accuracyOp;
}

void crossbowDataflowSetDataTransformOperator (crossbowDataflowP p, int id) {
	crossbowOperatorP dataTransformOp = crossbowDataflowFindKernel (p, id);
	invalidConditionException((! p->dataTransformOp));
	p->dataTransformOp = dataTransformOp;
}

int crossbowDataflowSize (crossbowDataflowP p) {
	return p->autoincrement;
}

void crossbowDataflowFree (crossbowDataflowP p) {
	if (! p)
		return;
	crossbowOperatorP temp;
	crossbowOperatorP curr = p->head->next;
	while (curr != p->tail) {
		temp = curr;
		curr = curr->next;
		crossbowOperatorFree (temp);
	}
	crossbowOperatorFree (p->head);
	crossbowOperatorFree (p->tail);
	crossbowFree (p, sizeof(crossbow_dataflow_t));
	return;
}

void crossbowDataflowDump (crossbowDataflowP p) {
	crossbowOperatorP op;
	char *s;
	printf ("=== [Dataflow %d: %d operators] ===\n", p->id, crossbowDataflowSize(p));
	op = p->head->next;
	while (op != p->tail) {
		s = crossbowKernelString (op->kernel);
		printf ("Op %d: %s\n", op->id, s);
		crossbowStringFree (s);
		op = op->next;
	}
	printf ("=== [End of dataflow dump] ===\n");
	fflush (stdout);
	return;
}

void crossbowDataflowDumpDependencyGraph (crossbowDataflowP p) {
	int i;
	crossbowOperatorP op;
	int size;
	printf ("=== [Dataflow %d: %d operators] ===\n", p->id, crossbowDataflowSize(p));
	op = p->head->next;
	while (op != p->tail) {
		printf ("Op %3d: branch %d\n", op->id, op->branch);
		if (! crossbowListEmpty(op->deps)) {
			size = crossbowListSize (op->deps);
			for (i = 0; i < size; ++i) {
				crossbowOperatorDependencyP dep = 
					(crossbowOperatorDependencyP) crossbowListPeek (op->deps, i);
				printf("\t%d must %s of %d\n", dep->guard->id, (dep->type == START_BEFORE_START) ? "start before start" : "end before start", op->id);
			}
		}
		op = op->next;
	}
	printf ("=== [End of dataflow dump] ===\n");
	fflush (stdout);
	return;
}
