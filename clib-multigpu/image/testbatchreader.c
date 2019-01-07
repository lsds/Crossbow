#include <stdio.h>
#include <stdlib.h>

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "image.h"
#include "record.h"
#include "recordreader.h"

#include "../timer.h"

#define USAGE "./testbatchreader [-d directory] [-s subset] [-n files] [-v level]"

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
 * Perform the kind of pre-processing for training images
 */
static void preprocessTrainingRecord (crossbowRecordP record) {

	crossbowImageCast (record->image);

	/*
	 * Sample bounding box:
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
	crossbowImageSampleDistortedBoundingBox (
		record->image, 
		record->boxes, 
		0.1, 
		&ratio[0],
		&area [0],
		100, 
		&height, &width, &top, &left);

	/* Crop image */
	crossbowImageCrop (record->image, height, width, top, left);

	/* Flip image */
	crossbowImageRandomFlipLeftRight (record->image);

	/* Resize image */
	crossbowImageResize (record->image, 224, 224);

	/* Distort image colours */
	crossbowImageMultiply (record->image, (1. / 255.));

	crossbowImageRandomBrightness (record->image, (32. / 255.));
	crossbowImageRandomContrast   (record->image, 0.5, 1.5);
	/* Lower saturation is 0.5, upper saturation is 1.5, max. delta hue is (0.2 x pi) */
	crossbowImageRandomHSVInYIQ   (record->image, 0.5, 1.5, 0.2 * 3.14);

	/* Clip by value */
	crossbowImageClipByValue (record->image, 0.0, 1.0);
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
    int batchsize = 32;
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
			batchsize = atoi(argv[j]);
		} else
		if (strcmp(argv[i], "-v") == 0) {
			verbose = (unsigned) atoi(argv[j]);
		} else {
			fprintf(stderr, "error: unknown flag: %s %s\n", argv[i], argv[j]);
		}
		i = j + 1;
	}
	
	/* Initialise memory manager */
	crossbowMemoryManagerInit ();
	
	crossbowRecordReaderP reader = crossbowRecordReaderCreate (workers);
	
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
    int buffersize = batchsize * 602112;
    void *buffer = (void *) crossbowMalloc (buffersize);
	
    crossbowTimerP timer = crossbowTimerCreate ();
    crossbowTimerStart (timer);
	int count = 0;
    /* Process 100 batches */
    while (count < 100) {
        crossbowRecordReaderRead (reader, batchsize, 602112, buffer, buffersize);
		count ++;
	}
    tstamp_t dt = crossbowTimerElapsedTime (timer);
    printf("%d batches (or %d images) processed\n", count, count * batchsize);
    printf("%llu usecs\n", dt);
    crossbowTimerFree (timer);
	crossbowRecordReaderFree (reader);
    crossbowFree(buffer, buffersize);
	crossbowMemoryManagerDump ();
	printf("Bye.\n");
	return 0;
}
