#include "list.h"

#include "memorymanager.h"

#include "utils.h"
#include "debug.h"

#include <time.h>

/* Return node to free list */
static void putNode (crossbowListP list, crossbowListNodeP p) {
   p->item = NULL;
   p->next = list->freeList;
   list->freeList = p;
}

static crossbowListNodeP crossbowListCreatePool (crossbowListP list, int size) {
	int i;
	crossbowListNodeP node;
	crossbowListNodePoolP pool;
	/* Create a new pool only when the list of free nodes is empty */
	invalidConditionException (list->freeList == NULL);
	/* Create a new stash of nodes */
	node = (crossbowListNodeP) crossbowMalloc (size * sizeof(crossbow_list_node_t));
	/* Create a new pool and initialise it */
	pool = (crossbowListNodePoolP) crossbowMalloc (sizeof(crossbow_list_node_pool_t));
	pool->node = node;
	pool->size = size;
	pool->next = list->pool;
	list->pool = pool;
	/* Append new nodes to free list */
	for (i = 0; i < pool->size; i++, node++)
		putNode (list, node);
	return list->freeList;
}

static crossbowListNodeP getNode (crossbowListP list) {
	crossbowListNodeP p = NULL;

	/* If `freeList` is NULL, all nodes are in use. */
	if (! (p = list->freeList)) {
		/* Create a new pool of nodes and append them to the `freeList` */
		p = crossbowListCreatePool (list, LIST_NODE_STASH);
	}
	list->freeList = p->next;
	return p;
}

crossbowListP crossbowListCreate () {
	crossbowListP p;
	p = (crossbowListP) crossbowMalloc (sizeof(crossbow_list_t));
	memset (p, 0, sizeof(crossbow_list_t));
	/* Initialise free list */
	crossbowListCreatePool (p, LIST_NODE_STASH);
	return p;
}

int crossbowListEmpty (crossbowListP p) {
	return (p->size == 0);
}

int crossbowListSize (crossbowListP p) {
	return (p->size);
}

void crossbowListAppend (crossbowListP p, void *item) {
	crossbowListNodeP n;
	n = getNode (p); /* (crossbowListNodeP) crossbowMalloc (sizeof(crossbow_list_node_t)); */
	n->item = item;
	n->next = NULL;
	if (crossbowListEmpty(p)) {
		p->head = n;
		p->tail = n;
	} else {
		p->tail->next = n;
		p->tail = n;
	}
	p->size++;
	return;
}

void crossbowListPrepend (crossbowListP p, void *item) {
	crossbowListNodeP n;
	n = getNode(p); /* (crossbowListNodeP) crossbowMalloc (sizeof(crossbow_list_node_t)); */
	n->item = item;
	n->next = NULL;
	if (crossbowListEmpty(p)) {
		p->head = n;
		p->tail = n;
	} else {
		n->next = p->head;
		p->head = n;
	}
	p->size++;
	return;
}

void *crossbowListPeek (crossbowListP p, int order) {
	crossbowListNodeP node;
	int ord;
	if (order == 0)
		return crossbowListPeekHead (p);
	/* Double-check that list is not empty */
	if (crossbowListEmpty(p))
		return NULL;
	node = p->head->next;
	ord = 1;
	while (node != NULL) {
		if (ord == order)
			return node->item;
		node = node->next;
		ord++;
	}
	return NULL;
}

void *crossbowListPeekHead (crossbowListP p) {
	if (crossbowListEmpty(p))
		return NULL;
	return p->head->item;
}

void *crossbowListPeekTail (crossbowListP p) {
	if (crossbowListEmpty(p))
		return NULL;
	return p->tail->item;
}

void *crossbowListRemoveFirst (crossbowListP p) {
	crossbowListNodeP node;
	void *item;
	if (crossbowListEmpty(p))
		return NULL;
	node = p->head;
	item = node->item;
	p->head = node->next;
	putNode(p, node);
	if (! --p->size)
		p->tail = NULL;
	/* crossbowFree (node, sizeof(crossbow_list_node_t)); */
	return item;
}

void crossbowListShuffle (crossbowListP p) {
	int i, j;
	void **array;
	void *t;
	int n;
	nullPointerException(p);
	if (crossbowListEmpty(p))
		return;
	n = crossbowListSize(p);
	if (n == 1)
		return;
	/* Convert list to array */
	array = crossbowMalloc (n * sizeof(void *));
	i = 0;
	while (! crossbowListEmpty(p)) {
		void *item = crossbowListRemoveFirst (p);
		array[i++] = item;
	}
	/* Shuffle array */
	srand(time(NULL));
	for (i = n - 1; i > 0; --i) {
		j = rand() % (i + 1);
		/* Swap item at position i with item at position j */
		t = array[i];
		array[i] = array[j];
		array[j] = t;
	}
	/* Re-populate the list */
	for (i = 0; i < n; ++i)
		crossbowListAppend(p, array[i]);
	/* Free temporal array */
	crossbowFree (array, (n * sizeof(void *)));
	return;
}

void crossbowListIteratorReset (crossbowListP p) {
	p->it = p->head;
	return;
}

unsigned crossbowListIteratorHasNext (crossbowListP p) {
	return (p->it != NULL);
}

void *crossbowListIteratorNext (crossbowListP p) {
	void *item = p->it->item;
	p->it = p->it->next;
	return item;
}

void crossbowListFree (crossbowListP p) {
	/* List must be empty. This ensure that items stored in list node are free'd as well. */
	if (! crossbowListEmpty(p))
		illegalOperationException();
	/* If not NULL, the `freeList` should contain at least one node. */
	nullPointerException(p->freeList);
	crossbowListNodeP last = p->freeList;
	int available = 1;
	while (last->next != NULL) {
		last = last->next;
		available++;
	}
	/* There exists at least one pool of nodes */
	nullPointerException (p->pool);
	crossbowListNodePoolP temp, pool = p->pool;
	int allocated = 0;
	while (pool != NULL) {
		temp = pool;
		pool = pool->next;
		/* Increment counter */
		allocated += temp->size;
		/* Free `temp` */
		crossbowFree (temp->node, temp->size * sizeof(crossbow_list_node_t));
		crossbowFree (temp, sizeof(crossbow_list_node_pool_t));
	}
	dbg("%2d/%2d nodes in pool\n", available, allocated);
	invalidConditionException(available == allocated);
	crossbowFree (p, sizeof(crossbow_list_t));
}
