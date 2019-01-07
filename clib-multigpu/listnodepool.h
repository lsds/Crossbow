#ifndef __CROSSBOW_LIST_NODE_POOL_H_
#define __CROSSBOW_LIST_NODE_POOL_H_

#include "listnode.h"

/*
 * Used by list.c and threadsafequeue.c
 */
typedef struct crossbow_list_node_pool *crossbowListNodePoolP;
typedef struct crossbow_list_node_pool {
	crossbowListNodeP node;
	int size;
	crossbowListNodePoolP next;
} crossbow_list_node_pool_t;

#endif /* __CROSSBOW_LIST_NODE_POOL_H_ */
