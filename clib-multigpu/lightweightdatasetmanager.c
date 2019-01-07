#include "lightweightdatasetmanager.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "datasetfile.h"

crossbowLightWeightDatasetManagerP crossbowLightWeightDatasetManagerCreate (int numberofnodes, unsigned GPU, int blocksize) {
	crossbowLightWeightDatasetManagerP p;
	p = (crossbowLightWeightDatasetManagerP) crossbowMalloc(sizeof(crossbow_lightweightdatasetmanager_t));

	p->registry = crossbowMemoryRegistryCreate (numberofnodes);

	p->GPU = GPU;

	p->blocksize = blocksize;
	p->padding   = 0;

	p->buffer = NULL;

	p->numberofslots = p->numberoftasks = 0;
	p->slots = NULL;
	p->tasks = NULL;

	return p;
}

void crossbowLightWeightDatasetManagerRegister (crossbowLightWeightDatasetManagerP p, int id, const char *filename) {
	crossbowMemoryRegistryNodeP node = crossbowMemoryRegistryGet (p->registry, id);
	nullPointerException(node);
	crossbowDatasetFileP file = crossbowDatasetFileCreate (filename);
	node->file = file;
	return;
}

void crossbowLightWeightDatasetManagerCreateSlots (crossbowLightWeightDatasetManagerP p, int numberofslots) {
	int id;
	crossbowLightWeightDatasetSlotP slot;

	nullPointerException (p);
	nullPointerException (p->tasks);

	p->numberofslots = numberofslots;
	invalidConditionException ((p->numberofslots * p->blocksize) == crossbowLightWeightDatasetBufferCapacity (p->buffer));

	/* Allocate slots */
	p->slots = (crossbowLightWeightDatasetSlotP) crossbowMalloc (p->numberofslots * sizeof(crossbow_lightweightdatasetslot_t));
	memset (p->slots, 0, (p->numberofslots * sizeof(crossbow_lightweightdatasetslot_t)));

	/* Initialise */
	for (id = 0; id < p->numberofslots; ++id) {
		/* Get pointer */
		slot = &(p->slots[id]);

		slot->id = id;

		slot->table = p->tasks;

		slot->ndx = id; /* Current task index; subsequent tasks are at offset `inc` are limited by `max` */
		slot->inc = p->numberofslots;
		slot->max = p->numberoftasks;

		slot->buffer = p->buffer;
		slot->offset = p->blocksize * id;
		slot->length = p->blocksize;
		/* For debugging & accounting purposes */
		slot->epochs = 0;
	}
}

void crossbowLightWeightDatasetManagerCreateTasks (crossbowLightWeightDatasetManagerP p, int batchsize, int elements, int fill) {
	int id;
	crossbowLightWeightDatasetTaskP task;

	crossbowMemoryRegistryNodeP node;

	int mark, remaining;

	int __tasksize; /* Task size, excluding padding (in bytes) */
	int __elemsize;

	nullPointerException (p);
	invalidConditionException ((elements % batchsize) == 0);

	p->numberoftasks = elements / batchsize;

	invalidConditionException (((p->blocksize - p->padding) % batchsize) == 0);

	__tasksize = (p->blocksize - p->padding);
	__elemsize = (p->blocksize - p->padding) / batchsize;

	p->tasks = (crossbowLightWeightDatasetTaskP) crossbowMalloc (p->numberoftasks * sizeof(crossbow_lightweightdatasettask_t));
	memset (p->tasks, 0, (p->numberoftasks * sizeof(crossbow_lightweightdatasettask_t)));

	/* Initialise */

	node = crossbowMemoryRegistryGet (p->registry, 0);
	nullPointerException (node);
	nullPointerException (node->file);

	mark = 0;
	remaining = crossbowDatasetFileSize (node->file);

	for (id = 0; id < p->numberoftasks; ++id) {
		/* Get pointer */
		task = &(p->tasks[id]);

		/* dbg("At task %010d: mark %10d remaining %10d file %s\n", id, mark, remaining, node->file->filename); */

		/* `remaining` is 0 when tasks fit perfectly within files. */
		invalidConditionException (remaining >= 0);
		if (remaining == 0) {
			/* There are no bytes remaining in the current file. Go to the next file and set `mark` to 0. */
			node = node->next;
			nullPointerException (node);
			mark = 0;
			remaining = crossbowDatasetFileSize (node->file);
		}
		task->file[0] = node->file;
		task->file[1] = NULL;
		task->offset  = mark;
		/* Set task length in bytes */
		task->length  = __tasksize;
		/* Increment mark */
		mark += __tasksize;
		/* Decrement bytes remaining in the current file. */
		remaining -= __tasksize;
		if (remaining < 0) {
			/* Current file buffer overflows. Go to the next file and set `mark` accordingly. */
			node = node->next;
			if (! node) {
				/* If there are no more files, fill the task with bytes from the first file. */
				invalidConditionException (id == (p->numberoftasks - 1));
				invalidConditionException ((-remaining) == (fill * __elemsize));
				node = crossbowMemoryRegistryGet (p->registry, 0);
			} else {
				nullPointerException (node);
			}
			/* Task spans across two files */
			task->file[1] = node->file;
			/* Set `mark` and `remaining`  */
			mark = -remaining;
			remaining = crossbowDatasetFileSize (node->file) + remaining;
		}
	}
	return;
}

void crossbowLightWeightDatasetManagerDump (crossbowLightWeightDatasetManagerP p) {
	int i;
	crossbowLightWeightDatasetSlotP slot;
	crossbowLightWeightDatasetTaskP task;

	printf ("=== [Light-weight dataset: %d slots, %d tasks] ===\n", p->numberofslots, p->numberoftasks);
	/* Dump slots */
	printf ("===\n");
	for (i = 0; i < p->numberofslots; ++i) {
		slot = &(p->slots[i]);
		/* Assuming that there are less than 1000 tasks in the queue
		 * and that the entire buffer is no more than 1 GB. */
		printf ("Slot %03d: buffer %p offset %10d length %10d\n",
				slot->id,
				slot->buffer->data,
				slot->offset,
				slot->length);
	}
	printf ("===\n");
	/* Dump tasks */
	for (i = 0; i < p->numberoftasks; ++i) {
		task = &(p->tasks[i]);
		if (task->file[1])
			printf ("Task %06d: files %p/%p offset %10d length %10d\n",
					i,
					task->file[0],
					task->file[1],
					task->offset,
					task->length);
		else
			printf ("Task %06d: files %p/%-14s offset %10d length %10d\n",
					i,
					task->file[0],
					"null",
					task->offset,
					task->length);
	}
	printf ("=== [End of light-weight dataset dump] ===\n");
	fflush (stdout);
	return;
}

void crossbowLightWeightDatasetManagerFree (crossbowLightWeightDatasetManagerP p) {
	if (! p)
		return;

	crossbowMemoryRegistryFree (p->registry, p->blocksize);
	crossbowLightWeightDatasetBufferFree (p->buffer, p->blocksize);

	crossbowFree (p->slots, (p->numberofslots * sizeof(crossbow_lightweightdatasetslot_t)));
	crossbowFree (p->tasks, (p->numberoftasks * sizeof(crossbow_lightweightdatasettask_t)));

	crossbowFree(p, sizeof(crossbow_lightweightdatasetmanager_t));

	return;
}
