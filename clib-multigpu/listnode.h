#ifndef __CROSSBOW_LIST_NODE_H_
#define __CROSSBOW_LIST_NODE_H_

/*
 * Used by list.c and threadsafequeue.c
 */
typedef struct crossbow_list_node *crossbowListNodeP;
typedef struct crossbow_list_node {
	void *item;
	crossbowListNodeP next;
} crossbow_list_node_t;

#endif /* __CROSSBOW_LIST_NODE_H_ */
