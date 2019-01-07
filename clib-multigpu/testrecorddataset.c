#include <stdio.h>
#include <stdlib.h>

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "timer.h"

#include "recorddataset.h"

#define USAGE "./testrecorddataset"

int main (int argc, char *argv[]) {

	(void) argc;
	(void) argv;

	crossbowRecordDatasetP dataset = NULL;
	int workers = 4;
	int capacity [2] = { 616562688, 131072 };
	int NB = 32;
	int b = 32;
	int padding [2] = { 0, 3968 };
	int files = 626;
	char *directory = "/data/crossbow/imagenet/ilsvrc2012/records";
	char *subset = "train";

	/* Initialise memory manager */
	crossbowMemoryManagerInit ();

	dataset = crossbowRecordDatasetCreate (workers, capacity, NB, b, padding);

	/* Register dataset */
	int idx;
	char filename[1024];
	for (idx = 0; idx < files; ++idx) {
		if (sprintf(filename, "%s/crossbow-%s.records.%d", directory, subset, (idx + 1)) < 0) {
			fprintf(stderr, "error: failed to generate filename\n");
			exit(1);
		}
		crossbowRecordReaderRegister (dataset->reader, filename);
	}

	info("Finalise dataset's reader\n");
	crossbowRecordReaderFinalise (dataset->reader);

	info("Fill dataset's buffer for the first time\n");

	crossbowRecordDatasetInitSafely (dataset);
    
    int count = 0, iterations = 1000000;
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
