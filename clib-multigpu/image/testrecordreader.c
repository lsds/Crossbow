#include <stdio.h>
#include <stdlib.h>

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "image.h"
#include "record.h"
#include "recordreader.h"

#include "../timer.h"

#include "yarng.h"

#define USAGE "./testrecordreader [-d directory] [-s subset] [-n files] [-v level]"

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
	
	return;
}

/*
 * Perform the kind of pre-processing for training images
 * that TensorFlow does in benchmarks/:
 *
 * https://github.com/alexandroskoliousis/benchmarks/blob/27b2ec139c86b39ab596321afe08878b36a5adfd/scripts/tf_cnn_benchmarks/preprocessing.py#L286
 */
static void preprocessTrainingRecord (crossbowRecordP record, int verbose) {

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
	
	if (verbose > 0)
		printf("Checksum of traning image is %.4f\n", crossbowImageChecksum (record->image));
	
}

int main (int argc, char *argv[]) {
	
	/* Input argument iterators */
	int i, j;
	/* Default input arguments */
	unsigned verbose = 0;
	char *directory = "examples";
	char *subset = "train";
	char *type = "test";
	int files = 1;
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
		if (strcmp(argv[i], "-t") == 0) {
			if ((strcmp (argv[j], "train") != 0) && (strcmp (argv[j], "test") != 0)) {
				fprintf(stderr, "error: invalid type '%s'. Try 'train' or 'test'\n", argv[j]);
				exit(1);
			}
			type = argv[j];
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
	
	crossbowYarngInit(123456789);
	
	crossbowRecordReaderP reader = crossbowRecordReaderCreate (1); /* 1 worker */
	
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
	
	crossbowTimerP timer = crossbowTimerCreate ();
	crossbowTimerStart (timer);
	int count = 0;
	/* Iterate over dataset */
	while (crossbowRecordReaderHasNext(reader) && (count < 10)) {
		crossbowRecordP record = crossbowRecordCreate ();
		crossbowRecordReaderNext (reader, record);
		if (verbose > 1) {
			char *s = crossbowRecordString(record);
			printf("%s\n", s);
			crossbowStringFree (s);
			/* crossbowImageDump (record->image, 5); */
		}
		if (strcmp (type, "train") == 0)
			preprocessTrainingRecord (record, verbose);
		else {
			preprocessTestRecord (record, verbose);
		}
		crossbowRecordFree (record);
		count ++;
	}
	tstamp_t dt = crossbowTimerElapsedTime (timer);
	printf("%d images processed\n", count);
	printf("%llu usecs\n", dt);
	crossbowTimerFree (timer);
	crossbowRecordReaderFree (reader);
	crossbowMemoryManagerDump ();
	printf("Bye.\n");
	return 0;
}

