#include "recordreader.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "../arraylist.h"

#include "../timer.h"

#include <pthread.h>

/*
 * Perform the kind of pre-processing for test images
 */
static void preprocessTestRecord (crossbowRecordP record, unsigned verbose) {

	crossbowImageCast (record->image);

	/* Resize image */

	float h = (float) crossbowImageInputHeight (record->image);
	float w = (float) crossbowImageInputWidth  (record->image);

	float factor = 1.15;

	float ratio = max(224. / h, 224. / w);

	int resizeheight = (int) (h * ratio * factor);
	int resizewidth  = (int) (w * ratio * factor);
	
	if (verbose > 0)
		printf("Resized image to (%d x %d)\n", resizeheight, resizewidth);

	crossbowImageResize (record->image, resizeheight, resizewidth);
	
	if (verbose > 0)
		printf("Checksum of resized image is %.4f\n", crossbowImageChecksum (record->image));

	/* Crop image */

	int top  = (resizeheight - 224) / 2;
	int left = (resizewidth  - 224) / 2;

	crossbowImageCrop (record->image, 224, 224, top, left);
	
	if (verbose > 0)
		printf("Checksum of cropped image is %.4f\n", crossbowImageChecksum (record->image));
	return;
}

/*
 * A worker thread
 */
static void *handle (void *args) {
    
    /*
     * The argument to each thread is an array list of tasks.
     * Handle it appropriately.
     */
    crossbowArrayListP list = (crossbowArrayListP) args;
    crossbowRecordReaderTaskP task = crossbowArrayListGet (list, 0);
    
    /* Pin thread to a particular core based on worker id */

    cpu_set_t set;
    int core = 2 + task->id;
    CPU_ZERO (&set);
    CPU_SET  (core, &set);
    sched_setaffinity (0, sizeof(set), &set);
    dbg("Decoder #%02d pinned on core %02d\n", task->id, core);
    
    /* Iterate over list of tasks */
    int idx;
    for (idx = 0; idx < crossbowArrayListSize (list); ++idx) {
        task = crossbowArrayListGet (list, idx);
        /* Create new record */
        crossbowRecordP record = crossbowRecordCreate ();
        /* Read record (thread-safe version) */
        crossbowRecordFileReadSafely (task->file, task->id, task->position, record);
        /* Pre-process record */
        preprocessTestRecord (record, 0);
        /* Copy decoded (augmented) image to buffer */
        crossbowImageCopy (record->image, task->buffer[0], task->offset[0], 0); /* Ignore limit */
        /* Copy label */
        if (task->buffer[1])
        	crossbowRecordLabelCopy (record, task->buffer[1], task->offset[1], 0); /* Ignore limit */
        /* Free record */
        crossbowRecordFree (record);
    }
    return args;
}

crossbowRecordReaderP crossbowRecordReaderCreate (int workers) {
    crossbowRecordReaderP p = NULL;
    p = (crossbowRecordReaderP) crossbowMalloc (sizeof(crossbow_record_reader_t));
    p->dataset = crossbowListCreate ();
    p->counter = 0;
    p->records = 0;
    p->limit = -1; /* Unlimited iterations over dataset */
    p->wraps =  0;
    p->current = NULL;
    p->finalised = 0;
    p->workers = workers;
    p->jc = 0;
    return p;
}

void crossbowRecordReaderCoreOffset (crossbowRecordReaderP p, int jc) {
	nullPointerException(p);
	p->jc = jc;
	return;
}

void crossbowRecordReaderRegister (crossbowRecordReaderP p, const char *filename) {
	nullPointerException(p);
	invalidConditionException (! (p->finalised));
	crossbowListAppend (p->dataset, crossbowRecordFileCreate (filename, p->workers));
	return;
}

void crossbowRecordReaderFinalise (crossbowRecordReaderP p) {

	int records;

	nullPointerException(p);
	invalidConditionException (! (p->finalised));

	/* Iterate over files in dataset and extract number of records */
	crossbowListIteratorReset (p->dataset);
	while (crossbowListIteratorHasNext(p->dataset)) {

		p->current = (crossbowRecordFileP) crossbowListIteratorNext (p->dataset);

		/* Read file header */
		records = crossbowRecordFileHeader (p->current);
		invalidConditionException(records > 0);
		/* Accumulate number of records */
		p->records += records;
	}
	invalidConditionException(p->records > 0);
	info("%d records in %d file%s\n", p->records, crossbowListSize(p->dataset), (crossbowListSize(p->dataset) > 1 ? "s" : ""));
	/* Reset file iterator */
	crossbowListIteratorReset (p->dataset);
	p->current = (crossbowRecordFileP) crossbowListIteratorNext (p->dataset);
	/* Finalise record reader */
	p->finalised = 1;
	return;
}

void crossbowRecordReaderRepeat (crossbowRecordReaderP p, int limit) {
	nullPointerException(p);
	invalidConditionException (! (p->finalised));
	invalidArgumentException (limit > 0);
	p->limit = limit;
}

unsigned crossbowRecordReaderHasNext (crossbowRecordReaderP p) {
	nullPointerException(p);
	invalidConditionException (p->finalised);

	/* Without a limit, there is always a next record to process */
	if (p->limit < 0)
		return 1;

	/* Limit is 1 or more epochs. The current epoch is: `p->wraps + 1` */
	if ((p->wraps + 1) < p->limit)
		return 1;

	/* The iterator is about to hit `p->limit`, so check if `current` is the last file */
	if (p->current != crossbowListPeekTail (p->dataset))
		return 1;

	/* Check if there are any bytes remaining in the current record file */
	if (crossbowRecordFileHasRemaining (p->current))
		return 1;

	/* Bye */
	return 0;
}

void crossbowRecordReaderNext (crossbowRecordReaderP p, crossbowRecordP record) {
	nullPointerException(p);
	invalidConditionException (p->finalised);

	if (! crossbowRecordFileHasRemaining (p->current)) {
		/* Reset current file */
		crossbowRecordFileReset (p->current, (crossbowListSize(p->dataset) != 1));
		if (! crossbowListIteratorHasNext (p->dataset)) {
			/* Reset file iterator */
			crossbowListIteratorReset (p->dataset);
			p->wraps ++;
		}
		p->current = crossbowListIteratorNext (p->dataset);
	}
	/* Read a record from `p->current` file */
	crossbowRecordFileRead (p->current, record);
    return;
}

crossbowRecordFileP crossbowRecordReaderNextPointer (crossbowRecordReaderP p, int *position) {
    nullPointerException(p);
    invalidConditionException (p->finalised);
    
    if (! crossbowRecordFileHasRemaining (p->current)) {
        /* Reset current file */
        crossbowRecordFileReset (p->current, (crossbowListSize(p->dataset) != 1));
        if (! crossbowListIteratorHasNext (p->dataset)) {
            /* Reset file iterator */
            crossbowListIteratorReset (p->dataset);
            p->wraps ++;
        }
        p->current = crossbowListIteratorNext (p->dataset);
    }
    /* Get next record position from `p->current` file */
    *position = crossbowRecordFileNextPointer (p->current);
    return p->current;
}

/*
 * Read `count` examples of size `size` into `buffer`
 */
void crossbowRecordReaderRead (crossbowRecordReaderP p, 
    int count, 
    int size,
    void *buffer,
    int limit) {
    
    int id;
    int counter;
    int ndx;
    int offset;
    int partition;
    
    crossbowRecordFileP file;
    int position;
    
    crossbowArrayListP list;
    crossbowRecordReaderTaskP task;
    
    nullPointerException (p);
    invalidConditionException (p->finalised);
    
    if (p->workers > 1) {
        
        /* Multi-threaded version */
        dbg("Decode %d examples with %d workers\n", count, p->workers);
        
        /* Create worker pool */
        pthread_t *pool = (pthread_t *) crossbowMalloc (p->workers * sizeof(pthread_t));
        
        /* Split work (number of tasks per worker) */
        invalidConditionException ((count % p->workers) == 0);
        partition = count / p->workers;
        
        /* Write offset for output buffer */
        offset  = 0;
        counter = 0;
        for (id = 0; id < p->workers; ++id) {
            
            /* Create list of tasks for worker */
            list = crossbowArrayListCreate (partition);
            
            /* Populate list */
            for (ndx = 0; ndx < partition; ++ndx) {
                /* Find next read pointer */
                file = crossbowRecordReaderNextPointer (p, &position);
                /* Create new task */
                task = crossbowMalloc (sizeof(crossbow_record_reader_task_t));
                /* Fill-in task */
                task->id = id;
                task->jc = p->jc;

                task->counter = (++counter);
                
                task->file = file;
                task->position = position;
                
                task->buffer[0] = buffer;
                task->buffer[1] = NULL;

                task->offset[0] = offset;
                task->offset[1] = 0;

                invalidConditionException ((offset + size) <= limit);
                
                /* Append to list */
                crossbowArrayListSet (list, ndx, task);
                
                /* Increment offset */
                offset += size;
            }
            
            /* Create worker thread */
            pthread_create (&(pool[id]), NULL, handle, (void *) list);
        }
        
        for (id = 0; id < p->workers; ++id) {
            /* Wait worker to finish */
            pthread_join(pool[id], (void **) &list);
            /* Free list of tasks for worker */
            for (ndx = 0; ndx < partition; ++ndx) {
                task = crossbowArrayListGet (list, ndx);
                crossbowFree (task, sizeof(crossbow_record_reader_task_t));
            }
            crossbowArrayListFree (list);
        }
        
        crossbowFree (pool, (p->workers * sizeof(pthread_t)));
    }
    else {
        /* Single-threaded version */
        
        offset  = 0;
        counter = 0;
        for (ndx = 0; ndx < count; ++ndx) {
            counter ++;
            /* Allocate new record */
            crossbowRecordP record = crossbowRecordCreate ();
            /* Read record and decode image therein */
            crossbowRecordReaderNext (p, record);
            /* Preprocess record */
            preprocessTestRecord (record, 0);
            /* Copy decoded image to buffer */
            offset += crossbowImageCopy (record->image, buffer, offset, limit);
            /* Free record */
            crossbowRecordFree (record);
        }
    }
}

void crossbowRecordReaderReadProperly (crossbowRecordReaderP p,
    int count,
    int *size,
	int b,
	int *padding,
    void *images,
	void *labels,
    int *limit) {

    int id;
    int counter;
    int ndx;
    int offset [2];

    int partition;

    crossbowRecordFileP file;
    int position;

    crossbowArrayListP list;
    crossbowRecordReaderTaskP task;

    crossbowTimerP timer = crossbowTimerCreate ();
    crossbowTimerStart (timer);

    nullPointerException (p);
    invalidConditionException (p->finalised);

    invalidConditionException (p->workers > 1);

	/* Create worker pool */
	pthread_t *pool = (pthread_t *) crossbowMalloc (p->workers * sizeof(pthread_t));

	/* Split work equally among workers (number of tasks per worker) */
	invalidConditionException ((count % p->workers) == 0);
	partition = count / p->workers;

	/* Ensure that each worker write one or more complete batches */
	invalidConditionException ((partition % b) == 0);

	/* Write offset for output buffer */
	offset[0] = offset[1] = 0;
	counter = 0;

	for (id = 0; id < p->workers; ++id) {

		/* Create list of tasks for worker */
		list = crossbowArrayListCreate (partition);

		/* Populate list */
		for (ndx = 0; ndx < partition; ++ndx) {

			/* Find next read pointer */
			file = crossbowRecordReaderNextPointer (p, &position);
			/* Create new task */
			task = crossbowMalloc (sizeof(crossbow_record_reader_task_t));
			/* Fill-in task */
			task->id = id;
			task->jc = p->jc;

			task->counter = (++counter);

			task->file = file;
			task->position = position;

			task->buffer[0] = images;
			task->buffer[1] = labels;

			task->offset[0] = offset [0];
			task->offset[1] = offset [1];

			dbg("Item #%04d: image offset %10d size %6d limit %10d\n", counter, offset[0], size[0], limit[0]);
			dbg("          : label offset %10d size %6d limit %10d\n",          offset[1], size[1], limit[1]);

			invalidConditionException ((offset[0] + size[0]) <= limit[0]);
			invalidConditionException ((offset[1] + size[1]) <= limit[1]);

			/* Append to list */
			crossbowArrayListSet (list, ndx, task);

			/* Increment offset */
			offset[0] += size[0] + ((counter % b == 0) ? padding[0] : 0);
			offset[1] += size[1] + ((counter % b == 0) ? padding[1] : 0);
		}

		/* Create worker thread */
		pthread_create (&(pool[id]), NULL, handle, (void *) list);
	}

	for (id = 0; id < p->workers; ++id) {
		/* Wait worker to finish */
		pthread_join(pool[id], (void **) &list);
		/* Free list of tasks for worker */
		for (ndx = 0; ndx < partition; ++ndx) {
			task = crossbowArrayListGet (list, ndx);
			crossbowFree (task, sizeof(crossbow_record_reader_task_t));
		}
		crossbowArrayListFree (list);
	}

	crossbowFree (pool, (p->workers * sizeof(pthread_t)));

	tstamp_t dt = crossbowTimerElapsedTime (timer);
	info("%d images processed in %llu usecs\n", count, dt);
	crossbowTimerFree (timer);

	return;

}

void crossbowRecordReaderFree (crossbowRecordReaderP p) {
    if (! p)
        return;
    if (p->dataset) {
    	while (! crossbowListEmpty(p->dataset)) {
    		crossbowRecordFileP file = crossbowListRemoveFirst (p->dataset); 
    		crossbowRecordFileFree (file);
    	}
    	crossbowListFree (p->dataset);
    }
    crossbowFree(p, sizeof(crossbow_record_reader_t));
}
