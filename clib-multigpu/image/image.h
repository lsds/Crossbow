#ifndef __CROSSBOW_IMAGE_H_
#define __CROSSBOW_IMAGE_H_

#include <stdio.h>
#include <stddef.h>
#include <jpeglib.h>

#include "../utils.h"

#include "../arraylist.h"

typedef struct jpeg_decompress_struct *crossbowJpegDecompressInfoP;

typedef struct jpeg_error_mgr *crossbowJpegErrorP;

typedef struct crossbow_image *crossbowImageP;
typedef struct crossbow_image {

	/* Decompressor state */
	crossbowJpegDecompressInfoP info;
	crossbowJpegErrorP err;

	/* Decoded image */
	unsigned char *img;
	int elements; /* Decoded image length (as reported in info) */

	/* Temporary buffer containing the transformed image */
	float *data;
	/* Current image height & width (based on transformations) */
	int channels, height, width;

	crossbowImageDataFormat_t format;

	unsigned isfloat; /* float representation */
	unsigned started;
	unsigned decoded;

	/* Expected output shape */
	int height_, width_;
	/* Pointer to output (double) buffer at given offset.
	 * Output is an array of floats.
	 */
	void *output;
	int offset;

} crossbow_image_t;

typedef struct crossbow_interpolation_weight *crossbowInterpolationWeightP;
typedef struct crossbow_interpolation_weight {

	long lower; /* Lower source index */
	long upper; /* Upper source index */
	float lerp; /* 1-D linear interpolation scale (lerp) */

} crossbow_interpolation_weight_t;

crossbowImageP crossbowImageCreate (int, int, int);

void crossbowImageReadFromMemory (crossbowImageP, void *, int);

void crossbowImageReadFromFile (crossbowImageP, FILE *);

void crossbowImageStartDecoding (crossbowImageP);

void crossbowImageDecode (crossbowImageP);

void crossbowImageCrop (crossbowImageP, int, int, int, int);

void crossbowImageCast (crossbowImageP);

int crossbowImageInputHeight (crossbowImageP);

int crossbowImageInputWidth (crossbowImageP);

int crossbowImageCurrentHeight (crossbowImageP);

int crossbowImageCurrentWidth (crossbowImageP);

int crossbowImageCurrentElements (crossbowImageP);

int crossbowImageOutputHeight (crossbowImageP);

int crossbowImageOutputWidth (crossbowImageP);

int crossbowImageChannels (crossbowImageP);

void crossbowImageRandomFlipLeftRight (crossbowImageP);

void crossbowImageSampleDistortedBoundingBox (crossbowImageP, crossbowArrayListP, float, float *, float *, int, int *, int *, int *, int *);

void crossbowImageResize (crossbowImageP, int, int);

void crossbowImageDistortColor (crossbowImageP);

void crossbowImageRandomBrightness (crossbowImageP, float);

void crossbowImageAdjustBrightness (crossbowImageP, float);

void crossbowImageRandomContrast (crossbowImageP, float, float);

void crossbowImageAdjustContrast (crossbowImageP, float);

void crossbowImageRandomSaturation (crossbowImageP);

void crossbowImageAdjustSaturation (crossbowImageP, float);

void crossbowImageRandomHue (crossbowImageP);

void crossbowImageAdjustHue (crossbowImageP, float);

void crossbowImageClipByValue (crossbowImageP, float, float);

void crossbowImageRandomHSVInYIQ (crossbowImageP, float, float, float);

void crossbowImageAdjustHSVInYIQ (crossbowImageP);

void crossbowImageMultiply (crossbowImageP, float);

void crossbowImageSubtract (crossbowImageP, float);

void crossbowImageCheckBounds (crossbowImageP, float, float);

char *crossbowImageString (crossbowImageP);

char *crossbowImageInfo (crossbowImageP);

double crossbowImageChecksum (crossbowImageP);

int crossbowImageCopy (crossbowImageP, void *, int, int);

void crossbowImageDump (crossbowImageP, int);

void crossbowImageDumpAsFloat (crossbowImageP, int);

void crossbowImageTranspose (crossbowImageP);

void crossbowImageFree (crossbowImageP);

#endif /* __CROSSBOW_IMAGE_H_ */
