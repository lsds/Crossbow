#include <stdio.h>
#include <stdlib.h>

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "image.h"
#include "record.h"
#include "recordreader.h"

#include "yarng.h"

#include "../timer.h"

#define USAGE "./testbatchreader [-d directory] [-s subset] [-n files] [-v level]"

/*
 * Perform the kind of pre-processing for test images
 */
static void preprocessTestRecord (crossbowRecordP record, unsigned verbose) {
	
	(void) record;
	(void) verbose;
	
	return;
}

/*
 * Perform the kind of pre-processing for training images
 */
static void preprocessTrainingRecord (crossbowRecordP record) {
	
	(void) record;
	
	return;
}

int main (int argc, char *argv[]) {
	
	/* Input argument iterators */
	int i, j;
	/* Default input arguments */
	unsigned verbose = 0;
	char *directory = "/mnt/nfs/users/piwatcha/my-tensorflow/data/imagenet/crossbow";
	char *subset = "train";
	int files = 1;
    int workers = 1;
    int b = 32;
	for (i = 1; i < argc;) {
		if ((j = i + 1) == argc) {
			fprintf(stderr, "usage: %s\n", USAGE);
			exit(1);
		}
		if (strcmp(argv[i], "-s") == 0) {
			if ((strcmp (argv[j], "train") != 0) && (strcmp (argv[j], "validation") != 0)) {
				fprintf(stderr, "error: invalid subset '%s'. Try 'train' or 'validation'\n", argv[j]);
				exit(1);
			}
			subset = argv[j];
		} else 
		if (strcmp(argv[i], "-d") == 0) {
			directory = argv[j];
		} else
		if (strcmp(argv[i], "-n") == 0) {
			files = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-w") == 0) {
			workers = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-b") == 0) {
			b = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-v") == 0) {
			verbose = (unsigned) atoi(argv[j]);
		} else {
			fprintf(stderr, "error: unknown flag: %s %s\n", argv[i], argv[j]);
		}
		i = j + 1;
	}
	
	(void) verbose;
	
	/* Initialise memory manager */
	crossbowMemoryManagerInit ();
	
	/* Initialise random number generator */
	crossbowYarngInit (123456789);
	
	crossbowRecordReaderP reader = crossbowRecordReaderCreate (workers, 0); /* no shuffle */
	
	/* Register dataset */
	int idx;
	char filename[1024];
	for (idx = 0; idx < files; ++idx) {
		if (sprintf(filename, "%s/crossbow-%s.records.%d", directory, subset, (idx + 1)) < 0) {
			fprintf(stderr, "error: failed to generate filename\n");
			exit(1);
		}
		crossbowRecordReaderRegister (reader, filename);
	}
	/* Repeat for only 1 epoch */
	crossbowRecordReaderRepeat (reader, 1);
	crossbowRecordReaderFinalise (reader);
    
    /* Create temporary buffer to hold batch of images. Every decoded 
     * image is (3 x 224 x 224) x sizeof(float) or 602,112 bytes long.
     */
    int buffersize = b * 602112;
    void *buffer = (void *) crossbowMalloc (buffersize);
	
    crossbowTimerP timer = crossbowTimerCreate ();
    crossbowTimerStart (timer);
	int count = 0;
    /* Process 100 batches */
    while (count < 100) {
        crossbowRecordReaderRead (reader, (strcmp (subset, "train") == 0) ? 1 : 0, b, 602112, buffer, buffersize);
		count ++;
	}
    tstamp_t dt = crossbowTimerElapsedTime (timer);
    printf("%d batches (or %d images) processed\n", count, count * b);
    printf("%llu usecs\n", dt);
    crossbowTimerFree (timer);
	crossbowRecordReaderFree (reader);
    crossbowFree(buffer, buffersize);
	crossbowMemoryManagerDump ();
	printf("Bye.\n");
	return 0;
}

