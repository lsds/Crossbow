#ifndef __CROSSBOW_LIST_H_
#define __CROSSBOW_LIST_H_

#include "listnode.h"

#include "listnodepool.h"

#define LIST_NODE_STASH 128

typedef struct crossbow_list *crossbowListP;
typedef struct crossbow_list {
	crossbowListNodeP head;
	crossbowListNodeP tail;
	int size;
	crossbowListNodeP freeList;
	crossbowListNodePoolP pool;
	crossbowListNodeP it;
} crossbow_list_t;

crossbowListP crossbowListCreate ();

int crossbowListEmpty (crossbowListP);
int crossbowListSize (crossbowListP);

void crossbowListAppend (crossbowListP, void *);
void crossbowListPrepend (crossbowListP, void *);

void *crossbowListPeek (crossbowListP, int);

void *crossbowListPeekHead (crossbowListP);
void *crossbowListPeekTail (crossbowListP);

void *crossbowListRemoveFirst (crossbowListP);

void crossbowListIteratorReset (crossbowListP);
unsigned crossbowListIteratorHasNext (crossbowListP);
void *crossbowListIteratorNext (crossbowListP);

void crossbowListFree (crossbowListP);

#endif /* __CROSSBOW_LIST_H_ */
