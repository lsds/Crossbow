#include "datasetfilehandler.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "memoryregion.h"

#include <pthread.h>

#define FREE_LIST_STASH 1024

static unsigned long autoincrement = 0UL;

static void *handle (void *args) {

	crossbowDatasetFileHandlerP self = NULL;
	crossbowDatasetFileBlockP  block = NULL;

	self = (crossbowDatasetFileHandlerP) args;

	if (self->offset > 0) {
		cpu_set_t set;
		int core = 18 + self->id; // self->offset + ((int) (self->id));
		CPU_ZERO (&set);
		CPU_SET  (core, &set);
		sched_setaffinity (0, sizeof(set), &set);
		info("Dataset handler #%lu pinned on core %d\n", self->id, core);
	}

	pthread_mutex_lock(&(self->lock));

	dbg("Dataset file handler #%lu starts\n", self->id);

	while (! self->exit) {
		/* Block, waiting for an event */
		while (!self->exit && crossbowListSize(self->events) == 0) {
			pthread_cond_wait(&(self->cond), &(self->lock));
		}
		if (self->exit) {
			break;
		}
		/* Get event */
		block = (crossbowDatasetFileBlockP) crossbowListRemoveFirst (self->events);

		if (block->op == 1) { /* Register block */

			if (block->file->region == NULL) {
				/* Operate on the file */
                
                dbg("Register block for file %s offset %10d length %10d\n",
                    block->file->filename, block->offset, block->length);
                
                if (block->gpu)
					crossbowDatasetFileRegisterRegion (block->file, block->offset, block->length);
				
                crossbowDatasetFileAdviceWillNeedRegion (block->file, block->offset, block->length);
			}
			else {

				dbg("Register region block for file %s offset %10d length %10d\n", 
                    block->file->filename, block->offset, block->length);

				/* First, copy the data block to the temporary memory region */
				crossbowMemoryRegionCopyBlock (block->file->region, block->offset, block->length, block->pad);
				
                /* Operate on the memory region */
                if (block->gpu)
					crossbowMemoryRegionRegister (block->file->region, block->offset, block->length);
				
                crossbowMemoryRegionAdviceWillNeed (block->file->region, block->offset, block->length);
			}
		}
		else { /* Unregister block */

			if (block->file->region == NULL) {
                /* Operate on the file */
				if (block->gpu)
					crossbowDatasetFileUnregisterRegion (block->file, block->offset, block->length);
				
                crossbowDatasetFileAdviceDontNeedRegion (block->file, block->offset, block->length);
			}
			else {

				dbg("Unregister region block for file %s offset %10d length %10d\n", 
                    block->file->filename, block->offset, block->length);

                /* Operate on the memory region */
				if (block->gpu)
					crossbowMemoryRegionUnregister (block->file->region, block->offset, block->length);
				
                crossbowMemoryRegionAdviceDontNeed (block->file->region, block->offset, block->length);
			}
		}

		/* Return block to free list */
		crossbowDatasetFileHandlerPutBlockSafely (self, block);
	}
	self->exited = 1;
	return self;
}

crossbowDatasetFileHandlerP crossbowDatasetFileHandlerCreate (int offset) {
	crossbowDatasetFileHandlerP p;
	p = (crossbowDatasetFileHandlerP) crossbowMalloc (sizeof(crossbow_datasetfilehandler_t));
	memset (p, 0, sizeof(crossbow_datasetfilehandler_t));
	p->exit = 0;
	p->exited = 0;
	p->id = autoincrement++;
	pthread_mutex_init (&(p->lock), NULL);
	pthread_cond_init  (&(p->cond), NULL);
	p->events = crossbowListCreate ();
	p->offset = offset;

	/* Manage free list */
	pthread_mutex_init (&(p->sync), NULL);
	/* Initialise free list */
	crossbowDatasetFileHandlerCreateBlockPool (p, FREE_LIST_STASH);

	/* Launch thread */
	pthread_create (&(p->thread), NULL, handle, (void *) p);
	return p;
}

crossbowDatasetFileBlockP crossbowDatasetFileHandlerGetBlock (crossbowDatasetFileHandlerP handler) {
	crossbowDatasetFileBlockP p = NULL;
	pthread_mutex_lock(&(handler->sync));
	if (! (p = handler->freeList)) {
		p = crossbowDatasetFileHandlerCreateBlockPool (handler, FREE_LIST_STASH);
	}
	handler->freeList = p->next;
	pthread_mutex_unlock(&(handler->sync));
	return p;
}

crossbowDatasetFileBlockP crossbowDatasetFileHandlerCreateBlockPool (crossbowDatasetFileHandlerP handler, int size) {
	int i;
	crossbowDatasetFileBlockP block;
	crossbowDatasetFileBlockPoolP pool;
	/* Create a new pool only when the list of free tasks is empty */
	invalidConditionException (handler->freeList == NULL);
	/* Create a new stash of tasks */
	block = (crossbowDatasetFileBlockP) crossbowMalloc (size * sizeof(crossbow_datasetfile_block_t));
	/* Create a new pool and initialise it */
	pool = (crossbowDatasetFileBlockPoolP) crossbowMalloc (sizeof(crossbow_datasetfile_block_pool_t));
	pool->block = block;
	pool->size = size;
	pool->next = handler->pool;
	handler->pool = pool;
	/* Append new nodes to free list */
	for (i = 0; i < pool->size; i++, block++)
		crossbowDatasetFileHandlerPutBlock (handler, block);
	return handler->freeList;
}

void crossbowDatasetFileHandlerPutBlock (crossbowDatasetFileHandlerP handler, crossbowDatasetFileBlockP p) {
	p->file = NULL;
	p->next = handler->freeList;
	handler->freeList = p;
	return;
}

void crossbowDatasetFileHandlerPutBlockSafely (crossbowDatasetFileHandlerP handler, crossbowDatasetFileBlockP p) {
	pthread_mutex_lock(&(handler->sync));
	p->file = NULL;
	p->next = handler->freeList;
	handler->freeList = p;
	pthread_mutex_unlock(&(handler->sync));
	return;
}

void crossbowDatasetFileHandlerPublish (crossbowDatasetFileHandlerP p, crossbowDatasetFileBlockP args) {
	pthread_mutex_lock (&(p->lock));
	if (! p->exit) {
		crossbowListAppend (p->events, args);
		pthread_cond_signal (&(p->cond));
	}
	pthread_mutex_unlock(&(p->lock));
}

void crossbowDatasetFileHandlerFree (crossbowDatasetFileHandlerP p) {
	pthread_mutex_lock(&(p->lock));
	if (! p->exited) {
		p->exit = 1;
		pthread_cond_signal (&(p->cond));
	}
	pthread_mutex_unlock(&(p->lock));
	/* Wait until thread has exited */
	pthread_join(p->thread, NULL);
	/* Free pool of nodes */
	nullPointerException(p->freeList);
	crossbowDatasetFileBlockP last = p->freeList;
	int available = 1;
	while (last->next != NULL) {
		last = last->next;
		available++;
	}
	/* There exists at least one pool of tasks */
	nullPointerException (p->pool);
	crossbowDatasetFileBlockPoolP temp, pool = p->pool;
	int allocated = 0;
	while (pool != NULL) {
		temp = pool;
		pool = pool->next;
		/* Increment counter */
		allocated += temp->size;
		/* Free `temp` */
		crossbowFree (temp->block, temp->size * sizeof(crossbow_datasetfile_block_t));
		crossbowFree (temp, sizeof(crossbow_datasetfile_block_pool_t));
	}
	dbg("%2d/%2d nodes in pool\n", available, allocated);
	invalidConditionException(available == allocated);
	/* List should be empty */
	crossbowListFree (p->events);
	crossbowFree (p, sizeof(crossbow_datasetfilehandler_t));
}
