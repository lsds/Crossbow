#include "recordreader.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "../arraylist.h"

#include "../timer.h"

#include <pthread.h>

/*
 * Perform the kind of pre-processing for test images
 * that TensorFlow does in benchmarks/:
 *
 * https://github.com/alexandroskoliousis/benchmarks/blob/27b2ec139c86b39ab596321afe08878b36a5adfd/scripts/tf_cnn_benchmarks/preprocessing.py#L198
 */
static void preprocessTestRecord (crossbowRecordP record, unsigned verbose) {
	
	/* Cast image to 32-bit float */
	crossbowImageCast (record->image);
	
	/* Get image height and width (and convert to floats) */
	float h = (float) crossbowImageInputHeight (record->image);
	float w = (float) crossbowImageInputWidth  (record->image);
	
	/* In ResNet, images are cropped to 256 x 256 and the final image size is 224 x 224.
	 * It is:
	 * 
	 * floor(224 x 1.45) ~= 256
	 */
	float factor = 1.145;
	
	/* Maintain aspect ratio */
	float ratio = max(224. / h, 224. / w);

	int resizeheight = (int) (h * ratio * factor);
	int resizewidth  = (int) (w * ratio * factor);
	
	if (verbose > 0)
		printf("Resized image to (%d x %d)\n", resizeheight, resizewidth);
	
	/* Resize the image to shape using the bilinear method (do not align corners) */
	crossbowImageResize (record->image, resizeheight, resizewidth);
	
	if (verbose > 0)
		printf("Checksum of resized image is %.4f\n", crossbowImageChecksum (record->image));
	
	/* Crop image to size (224, 224) */
	int top  = floor((float) (resizeheight - 224) / 2.); /* x // y */
	int left = floor((float) (resizewidth  - 224) / 2.);
	
	crossbowImageCrop (record->image, 224, 224, top, left);
	
	if (verbose > 0)
		printf("Checksum of cropped image is %.4f\n", crossbowImageChecksum (record->image));
	
	/* Rescale from [0, 255] to [0, 2] */
	crossbowImageMultiply(record->image, 1. / 127.5);
	/* Rescale to [-1, 1] */
	crossbowImageSubtract(record->image, 1.);
	
	return;
}

/*
 * Perform the kind of pre-processing for training images
 * that TensorFlow does in benchmarks/:
 *
 * https://github.com/alexandroskoliousis/benchmarks/blob/27b2ec139c86b39ab596321afe08878b36a5adfd/scripts/tf_cnn_benchmarks/preprocessing.py#L286
 */
static void preprocessTrainingRecord (crossbowRecordP record, int verbose) {
	
	(void) verbose;
	
	/* Cast image to 32-bit float */
	crossbowImageCast (record->image);

	/*
	 * Sample bounding box. If not box is supplied,
	 * assume the bounding box is the entire image.
	 *
	 * Minimum coverage is 0.1
	 * Aspect ratio range is [0.75, 1.33]
	 * Area range is [0.05, 1.0]
	 * Max. attempts is 100
	 */
	int height = 0;
	int width  = 0;
	int top    = 0;
	int left   = 0;

	float ratio [2] = {0.75, 1.33};
	float area  [2] = {0.05, 1.00};
	
	dbg("Sample bounding box\n");
	crossbowImageSampleDistortedBoundingBox (
		record->image, 
		record->boxes, 
		0.1, 
		&ratio[0],
		&area [0],
		100, 
		&height, &width, &top, &left);

	/* Crop image to the specified bounding box */
	crossbowImageCrop (record->image, height, width, top, left);

	/* Flip image */
	dbg("Flip image\n");
	crossbowImageRandomFlipLeftRight (record->image);

	/* Resize image to shape (224, 224) with the bilinear method (don't align corners) */
	dbg("Crop image to (224 x 224)\n");
	crossbowImageResize (record->image, 224, 224);
	
	/* Rescale from [0, 255] to [0, 2] */
	crossbowImageMultiply(record->image, 1. / 127.5);
	/* Rescale to [-1, 1] */
	crossbowImageSubtract(record->image, 1.);
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
    int core = 18 + task->id;
    CPU_ZERO (&set);
    CPU_SET  (core, &set);
    sched_setaffinity (0, sizeof(set), &set);
    dbg("Decoder #%02d pinned on core %02d\n", task->id, core);
    
	int bytes = 0;
    /* Iterate over list of tasks */
    int idx;
    for (idx = 0; idx < crossbowArrayListSize (list); ++idx) {
        task = crossbowArrayListGet (list, idx);
        /* Create new record */
        crossbowRecordP record = crossbowRecordCreate ();
        /* Read record (thread-safe version) */
        crossbowRecordFileReadSafely (task->file, task->id, task->position, record);
		bytes += record->length;
        /* Pre-process record */
		if (task->training) {
			preprocessTrainingRecord (record, 0);
		} else {
			preprocessTestRecord (record, 0);
		}
        /* Copy decoded (augmented) image to buffer */
        crossbowImageCopy (record->image, task->buffer[0], task->offset[0], 0); /* Ignore limit */
        /* Copy label */
        if (task->buffer[1])
        	crossbowRecordLabelCopy (record, task->buffer[1], task->offset[1], 0); /* Ignore limit */
        /* Free record */
        crossbowRecordFree (record);
    }
    dbg("Decoder processed %d bytes\n", bytes);
    return args;
}

crossbowRecordReaderP crossbowRecordReaderCreate (int workers) {
    crossbowRecordReaderP p = NULL;
    p = (crossbowRecordReaderP) crossbowMalloc (sizeof(crossbow_record_reader_t));
	p->shuffle = 1;
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
			/* Shuffle the files */
			if (p->shuffle)
				crossbowListShuffle (p->dataset);
			/* Reset file iterator */
			crossbowListIteratorReset (p->dataset);
			p->wraps ++;
			info("Wrap #%03d\n", p->wraps);
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
			/* Shuffle the files */
			if (p->shuffle)
				crossbowListShuffle (p->dataset);
            /* Reset file iterator */
            crossbowListIteratorReset (p->dataset);
            p->wraps ++;
			info("Wrap #%03d\n", p->wraps);
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
	unsigned training,
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
				task->training = training;
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
	unsigned training,
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

    invalidConditionException (p->workers > 0);

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
			task->training = training;
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
	double throughput = (((double) count) * 1000000.0) / ((double) dt);
	info("%6d images processed in %7llu usecs: %7.1f images/sec\n", count, dt, throughput);
	crossbowTimerFree (timer);

	return;
}

void crossbowRecordReaderFree (crossbowRecordReaderP p) {
    if (! p)
        return;
    if (p->dataset) {
    	while (! crossbowListEmpty(p->dataset)) {
    		crossbowRecordFileP file = crossbowListRemoveFirst (p->dataset); 
			/* info("Free record file %s\n", file->filename); */
    		crossbowRecordFileFree (file);
    	}
		/* info("Free list of files\n"); */
    	crossbowListFree (p->dataset);
    }
    crossbowFree(p, sizeof(crossbow_record_reader_t));
}
