#include <stdio.h>
#include <stdlib.h>

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "timer.h"

#include "recorddataset.h"

#include "image/yarng.h"

#define USAGE "./testrecorddataset [-sdnwbv...]"

int main (int argc, char *argv[]) {

	crossbowRecordDatasetP dataset = NULL;
	
	/* The number of batches pre-processed at a time */
	int NB = 32;
	/* The batch size */
	int b = 32;
	/* The number of pre-processing threads */
	int workers = 1;
	/* The number of files in the dataset */
	int files = 626;
	/* The location of the dataset */
	char *directory = "/data/crossbow/imagenet/records";
	/* The kind of dataset (train or validation) */
	char *subset = "train";
	/* Number of iterations */
	int iterations = 100;
	
	int i, j;
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
		if (strcmp(argv[i], "-f") == 0) {
			files = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-w") == 0) {
			workers = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-b") == 0) {
			b = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-n") == 0) {
			NB = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-i") == 0) {
			iterations = atoi(argv[j]);
		} else {
			fprintf(stderr, "error: unknown flag: %s %s\n", argv[i], argv[j]);
		}
		i = j + 1;
	}
	
	int padding [2] = { 0, 0 };
	/* 
	 * Calculate padding per batch for images and labels
	 * so that each batch is page-aligned.
	 * 
	 * Each image is 602112 bytes long.
	 * Each label is 4 bytes long.
	 */
	padding[0] = (((b * 602112) % 4096) == 0) ? 0 : (4096 - ((b * 602112) % 4096));
	padding[1] = (((b *      4) % 4096) == 0) ? 0 : (4096 - ((b *      4) % 4096));
	
	invalidConditionException ((((b * 602112) + padding[0]) % 4096) == 0);
	invalidConditionException ((((b *      4) + padding[1]) % 4096) == 0);
	
	/* Calculate the capacity of temporal buffers */
	int capacity [2] = { 0, 0 };
	
	capacity[0] = NB * (padding[0] + (b * 602112));
	capacity[1] = NB * (padding[1] + (b *      4));
	
	/* Initialise memory manager */
	crossbowMemoryManagerInit ();
	
	/* Initialise (yet another) random number generator */
	crossbowYarngInit (123456789);
	
	/* Create dataset */
	dataset = crossbowRecordDatasetCreate (workers, capacity, NB, b, padding, (strcmp(subset, "train") == 0) ? TRAIN : CHECK);

	/* Register dataset files with the record reader */
	int idx;
	char filename[1024];
	for (idx = 0; idx < files; ++idx) {
		if (sprintf(filename, "%s/crossbow-%s.records.%d", directory, subset, (idx + 1)) < 0) {
			fprintf(stderr, "error: failed to generate filename\n");
			exit(1);
		}
		crossbowRecordReaderRegister (dataset->reader, filename);
	}
	/* Finalise record reader */
	crossbowRecordReaderFinalise (dataset->reader);

	info("Fill dataset's buffer for the first time\n");
	crossbowRecordDatasetInitSafely (dataset);
    
    int count = 0;
    while (count < iterations) {
        /* Swap buffers (assuming instant processing) */
        crossbowRecordDatasetSwap (dataset);
        count ++;
    }
    printf("Done after %d iterations.\n", count);
    
	crossbowRecordDatasetFree (dataset);
	crossbowMemoryManagerDump ();
	printf("Bye.\n");
	return 0;
}
