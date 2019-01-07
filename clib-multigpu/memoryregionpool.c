#include "memoryregionpool.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowMemoryRegionPoolP crossbowMemoryRegionPoolCreate (int capacity) {
	crossbowMemoryRegionPoolP p;
	p = (crossbowMemoryRegionPoolP) crossbowMalloc (sizeof(crossbow_memory_region_pool_t));
	p->size = 0;
    p->autoincrement = 0;
	p->capacity = capacity;
	pthread_mutex_init (&(p->sync), NULL);
	p->list = NULL;
	return p;
}

crossbowMemoryRegionP crossbowMemoryRegionPoolGet (crossbowMemoryRegionPoolP p) {
	crossbowMemoryRegionP q = NULL;
	pthread_mutex_lock(&(p->sync));
	if (! (q = p->list)) {
		/* Create new memory region */
		q = crossbowMemoryRegionCreate (p->autoincrement++, p->capacity, (void *) p);
        info("Created memory region #%03d at %p\n", q->id, q->data);
	} else {
		p->list = q->next;
		p->size --;
        info("Reusing memory region #%03d at %p\n", q->id, q->data);
	}
	pthread_mutex_unlock(&(p->sync));
	return q;
}

void crossbowMemoryRegionPoolRelease (crossbowMemoryRegionPoolP p, crossbowMemoryRegionP q) {
	pthread_mutex_lock(&(p->sync));
	crossbowMemoryRegionReset (q);
	q->next = p->list;
	p->list = q;
	p->size ++;
	pthread_mutex_unlock(&(p->sync));
}

void crossbowMemoryRegionPoolFree (crossbowMemoryRegionPoolP p) {
	if (! p)
		return;
	crossbowMemoryRegionP q;
    int count = 0;
	while (p->size > 0) {
		q = crossbowMemoryRegionPoolGet (p);
		crossbowMemoryRegionFree (q);
        count ++;
	}
    /* Did we free all allocated memory regions? */
    invalidConditionException (count == p->autoincrement);
    /* Free the pool */
	crossbowFree (p, sizeof(crossbow_memory_region_pool_t));
	return;
}
