#include "image.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "rectangle.h"
#include "boundingbox.h"

#include "yarng.h"

crossbowImageP crossbowImageCreate (int channels, int height, int width) {
	crossbowImageP p = NULL;
	p = (crossbowImageP) crossbowMalloc (sizeof(crossbow_image_t));
	memset(p, 0, sizeof(crossbow_image_t));

	p->channels = channels;

	/* Configure output image shape */
	p->height_ = height;
	p->width_ = width;

	/*
	 * Allocate a new decompress struct
	 * with the default error handler.
	 */
	p->info = (crossbowJpegDecompressInfoP) crossbowMalloc (sizeof(struct jpeg_decompress_struct));
	p->err = (crossbowJpegErrorP) crossbowMalloc (sizeof(struct jpeg_error_mgr));
	p->info->err = jpeg_std_error (p->err);
	jpeg_create_decompress (p->info);

	p->data = NULL;
	p->height = p->width = 0;

	p->format = HWC;

	p->isfloat = 0; /* Float representation of the image (stored in `data`) */
	p->started = 0;
	p->decoded = 0;

	p->output = NULL;
	p->offset = 0;
	return p;
}

void crossbowImageReadFromMemory (crossbowImageP p, void *data, int length) {
	nullPointerException(p);
	jpeg_mem_src (p->info, data, length);
	return;
}

void crossbowImageReadFromFile (crossbowImageP p, FILE *file) {
	nullPointerException(p);
	jpeg_stdio_src (p->info, file);
	return;
}

void crossbowImageStartDecoding (crossbowImageP p) {
	nullPointerException(p);
	/* Read file header, set default decompression parameters */
	if (jpeg_read_header (p->info, TRUE) != 1)
		err("Failed to read JPEG header\n");
	/* Try integer fast... */
	p->info->dct_method = JDCT_IFAST;
	/* p->info->out_color_space = JCS_RGB; */
	if (p->info->jpeg_color_space == JCS_CMYK || p->info->jpeg_color_space == JCS_YCCK) {
		/* Always use CMYK for output in a 4 channel JPEG. The library 
		 * has a builtin decoder. We will later convert to RGB.
		 */
		info("Set colour space to CMYK\n");
		p->info->out_color_space = JCS_CMYK;
	} else {
		p->info->out_color_space = JCS_RGB;
	}
	jpeg_start_decompress(p->info);
	if (p->info->out_color_space == JCS_CMYK) {
		info("JPEG image (%d x %d x %d)\n", 
			p->info->output_height, p->info->output_width, p->info->output_components);
	}
	/* 
	 * JPEG images must have 3 channels. But we need to account for CMYK images.
	 * invalidConditionException (p->info->output_components == p->channels);
	 */
	p->started = 1;
	return;
}

void crossbowImageDecode (crossbowImageP p) {
	nullPointerException(p);
	/* Is `p->info` filled? */
	invalidConditionException (p->started);
	unsigned CMYK = (p->info->out_color_space == JCS_CMYK);
	if (p->decoded)
		return;
	p->elements = crossbowImageInputHeight (p) * crossbowImageInputWidth (p) * crossbowImageChannels (p);
	if (CMYK) {
		info("Allocate an image buffer of size (%d x %d x %d)\n",
			crossbowImageInputHeight (p),
			crossbowImageInputWidth  (p),
			crossbowImageChannels    (p)
		);
	}
	p->img = (unsigned char *) crossbowMalloc (p->elements);
	
	unsigned char *temp[1];
	temp[0] = NULL;
	if (CMYK) {
		/* Allocate temporal buffer */
		info("Allocate a temporal buffer of (%d x %d)\n", p->info->output_width, p->info->output_components);
		temp[0] = (unsigned char *) crossbowMalloc (p->info->output_width * p->info->output_components);
	}
	
	/* Decompress */

	/* Compute row stride */
	int stride = crossbowImageInputWidth (p) * crossbowImageChannels (p);
	unsigned char *buffer[1];

	/* By default, scanlines will come out in RGBRGBRGB...  order */
	int lines = 0;
	while (p->info->output_scanline < (unsigned int) crossbowImageInputHeight (p)) {
		/* Set pointer to image buffer */
		buffer[0] = p->img + p->info->output_scanline * stride;
		if (CMYK) {
			/* Read into temporal buffer */
			lines += jpeg_read_scanlines(p->info, temp, 1);
			/* Convert CMYK to RGB */
			for (int i = 0; i < crossbowImageInputWidth (p); ++i) {
				int offset = 4 * i;
				const int c = (int) temp[0][offset + 0];
				const int m = (int) temp[0][offset + 1];
				const int y = (int) temp[0][offset + 2];
				const int k = (int) temp[0][offset + 3];
				int r, g, b;
				if (p->info->saw_Adobe_marker) {
					r = (k * c) / 255;
					g = (k * m) / 255;
					b = (k * y) / 255;
				} else {
					r = (255 - k) * (255 - c) / 255;
					g = (255 - k) * (255 - m) / 255;
					b = (255 - k) * (255 - y) / 255;
				}
				buffer[0][3 * i + 0] = r;
				buffer[0][3 * i + 1] = g;
				buffer[0][3 * i + 2] = b;
			}
		} else {
			/* Set pointer to image buffer */
			buffer[0] = p->img + p->info->output_scanline * stride;
			jpeg_read_scanlines(p->info, buffer, 1);
		}
	}
	if (CMYK) {
		info("%d lines read\n", lines);
	}
	
	if (CMYK) {
		crossbowFree (temp[0], p->info->output_width * p->info->output_components);
	}
	
	jpeg_finish_decompress(p->info);

	p->decoded = 1;
	return;
}

void crossbowImageCast (crossbowImageP p) {
	int i;
	nullPointerException (p);
	/* Is the image decoded? */
	invalidConditionException (p->decoded);
	if (p->isfloat)
		return;
	p->data = (float *) crossbowMalloc (p->elements * sizeof(float));
	for (i = 0; i < p->elements; i++)
		p->data[i] = (float) p->img[i];
	p->isfloat = 1;
	/* Set current height & width */
	p->height = crossbowImageInputHeight (p);
	p->width  = crossbowImageInputWidth  (p);
	return;
}

void crossbowImageTranspose (crossbowImageP p) {
    int c, h, w;
    nullPointerException (p);
	invalidConditionException (p->decoded);
	invalidConditionException (p->isfloat);
    /* Allocate new buffer */
    float *transposed = (float *) crossbowMalloc (crossbowImageCurrentElements (p) * sizeof(float));
    int src, dst;
    for (h = 0; h < p->height; ++h) {
        for (w = 0; w < p->width; ++w) {
            for (c = 0; c < p->channels; ++c) {
                src = ((h * p->width  + w) * p->channels + c);
                dst = ((c * p->height + h) * p->width    + w);
                transposed[dst] = p->data[src];
            }
        }
    }
    
    /* Free current data pointer */
    crossbowFree (p->data, crossbowImageCurrentElements (p) * sizeof(float));
    /* Assign new data pointer */
    p->data = transposed;
    return;
}

int crossbowImageInputHeight (crossbowImageP p) {
	nullPointerException(p);
	invalidConditionException (p->started);
	return p->info->output_height;
}

int crossbowImageInputWidth (crossbowImageP p) {
	nullPointerException(p);
	invalidConditionException (p->started);
	return p->info->output_width;
}

int crossbowImageCurrentHeight (crossbowImageP p) {
	nullPointerException(p);
	invalidConditionException (p->isfloat);
	return p->height;
}

int crossbowImageCurrentWidth (crossbowImageP p) {
	nullPointerException(p);
	invalidConditionException (p->isfloat);
	return p->width;
}

int crossbowImageCurrentElements (crossbowImageP p) {
	return (crossbowImageCurrentHeight (p) *
			crossbowImageCurrentWidth  (p) * crossbowImageChannels (p));
}

int crossbowImageOutputHeight (crossbowImageP p) {
	nullPointerException(p);
	return p->height_;
}

int crossbowImageOutputWidth (crossbowImageP p) {
	nullPointerException(p);
	return p->width_;
}

int crossbowImageChannels (crossbowImageP p) {
	nullPointerException(p);
	return p->channels;
}

void crossbowImageCrop (crossbowImageP p, int height, int width, int top, int left) {
	int x, y;
	int start, offset = 0;
	float  *input;
	float *output;

	nullPointerException(p);
	invalidConditionException (p->isfloat);

	/*
	dbg("Crop image from (%d x %d) to (%d x %d) starting at (%d, %d)\n",
			crossbowImageCurrentHeight (p),
			crossbowImageCurrentWidth  (p),
			height,
			width,
			top, left);
	*/

	/* Allocate new buffer */
	float *cropped = (float *) crossbowMalloc (crossbowImageChannels (p) * height * width * sizeof(float));

	/* Compute new row size */
	int rowsize = width * crossbowImageChannels (p);

	for (y = 0; y < height; ++y) {
		/* Move input data pointer to top-left position */
		start = ((top + y) * crossbowImageChannels (p) * crossbowImageCurrentWidth (p)) + (left * crossbowImageChannels (p));
		input = p->data + start;
		/* Copy `width` pixels to output (a complete row) */
		output = cropped + offset;
		for (x = 0; x < rowsize; ++x)
			output[x] = input[x];
		/* Move output data pointer */
		offset += rowsize;
	}
	/* Free current data pointer */
	crossbowFree (p->data, crossbowImageCurrentElements (p) * sizeof(float));
	/* Assign new data pointer */
	p->data = cropped;
	/* Set current data, height & width */
	p->height = height;
	p->width  =  width;
	return;
}

/* 
 * With a 50% chance, outputs the contents of `image` flipped along the width. 
 * Otherwise output the image as-is.
 */
void crossbowImageRandomFlipLeftRight (crossbowImageP p) {
	int c, h, w;
	
	nullPointerException (p);
	invalidConditionException (p->decoded);
	invalidConditionException (p->isfloat);
	
	int height = crossbowImageCurrentHeight (p);
	int width = crossbowImageCurrentWidth (p);
	int channels = p->channels;
	
	/* Flip a coin: if value is less that 0.5, mirror the image */
	float value = crossbowYarngNext (0.0, 1.0);
	unsigned mirror = (value < 0.5F);
	if (! mirror) return;
	dbg("Flip image (%d x %d x %d)\n", height, width, channels);
	int src, dst;
	float tmp;
	for (h = 0; h < height; ++h) {
		for (w = 0; w < width; ++w) {
			for (c = 0; c < channels; ++c) {
				src = ((h * width  + w) * channels + c);
				dst = ((h * width  + (width - 1 - w)) * channels + c);
				tmp = p->data[src];
				p->data[src] = p->data[dst];
				p->data[dst] = tmp;
			}
		}
	}
	return;
}

void crossbowImageDistortColor (crossbowImageP p) {
	nullPointerException(p);
	return;
}

void crossbowImageRandomBrightness (crossbowImageP p, float delta) {
	nullPointerException(p);
	invalidArgumentException(delta > 0);

	float brightness = crossbowYarngNext (-delta, delta);
	crossbowImageAdjustBrightness (p, brightness);

	return;
}

/**
 * Requires float representation.
 *
 * The value `delta` is added to all components of the image. 
 * `delta` should be in the range `[0,1)`. Pixels should also 
 * be in that range.
 */
void crossbowImageAdjustBrightness (crossbowImageP p, float delta) {
	nullPointerException(p);
	invalidConditionException(p->isfloat);
	invalidArgumentException(((delta >= 0) && (delta < 1)));
	int i;
	int elements = crossbowImageCurrentElements (p);
	for (i = 0; i < elements; ++i)
		p->data[i] += delta;
	return;
}

void crossbowImageRandomContrast (crossbowImageP p, float lower, float upper) {
	nullPointerException(p);
	invalidArgumentException(lower >= 0);
	invalidArgumentException(lower <= upper);

	float contrast = crossbowYarngNext (lower, upper);
	crossbowImageAdjustContrast (p, contrast);
	return;
}

/**
 * Requires float representation.
 * 
 * Contrast is adjusted independently for 
 * each channel of each image.
 * 
 * For each channel, the function computes 
 * the mean of the pixels in the channel.
 * 
 * It then adjusts each component `x` of a
 * pixel:
 *    x = (x - mean) * factor + mean
 */
void crossbowImageAdjustContrast (crossbowImageP p, float factor) {
	nullPointerException(p);
	invalidConditionException(p->isfloat);
	/* Compute mean per channel */
	int i;
	float R, G, B;
	float N = (float) (crossbowImageCurrentHeight (p) * crossbowImageCurrentWidth (p));
	int elements = crossbowImageCurrentElements (p);
	R = G = B = 0;
	for (i = 0; i < elements; i += p->channels) {
		R += p->data[i + 0];
		G += p->data[i + 1];
		B += p->data[i + 2];
	}
	R /= N;
	G /= N;
	B /= N;
	/* Adjust contrast by factor */
	for (i = 0; i < elements; i += p->channels) {
		p->data[i + 0] = (p->data[i + 0] - R) * factor + R;
		p->data[i + 1] = (p->data[i + 1] - G) * factor + G;
		p->data[i + 2] = (p->data[i + 2] - B) * factor + B;
	}
	return;
}

void crossbowImageRandomSaturation (crossbowImageP p) {
	crossbowImageAdjustSaturation (p, 0);
	return;
}

/**
 * Requires float representation
 *
 * The image saturation is adjusted by  converting 
 * the image to HSV and multiplying the saturation 
 * channel (S) by `factor` and clipping. 
 * 
 * The image is then converted back to RGB.
 */
void crossbowImageAdjustSaturation (crossbowImageP p, float factor) {
	nullPointerException(p);
	(void) factor;
	unsupportedOperationException ();
	return;
}

void crossbowImageRandomHue (crossbowImageP p) {
	crossbowImageAdjustHue (p, 0);
	return;
}

/**
 * Requires float representation.
 * 
 * image` is an RGB image.  
 * The image hue is adjusted by converting the image 
 * to HSV and rotating the hue channel (H) by `delta`.  
 * 
 * The image is then converted back to RGB.
 */
void crossbowImageAdjustHue (crossbowImageP p, float delta) {
	nullPointerException(p);
	(void) delta;
	unsupportedOperationException ();
	return;
}

/**
 * Requires float representation.
 *
 * Any value less that `lower` is set to `lower`; and
 * any value greater than `upper` is set to `upper`.
 */
void crossbowImageClipByValue (crossbowImageP p, float lower, float upper) {
	int i;
	int elements;
	nullPointerException(p);
	invalidConditionException(p->isfloat);
	elements = crossbowImageCurrentElements (p);
	for (i = 0; i < elements; ++i) {
		if (p->data [i] < lower)
			p->data [i] = lower;
		if (p->data [i] > upper)
			p->data [i] = upper;
	}
	return;
}

/**
 * Notes:
 *
 * lower: Lower saturation
 * upper: Upper saturation
 * delta: Delta HUE
 */
void crossbowImageRandomHSVInYIQ (crossbowImageP p, float lower, float upper, float delta) {
	nullPointerException(p);
	(void) lower;
	(void) upper;
	(void) delta;
	return;
}

void crossbowImageAdjustHSVInYIQ (crossbowImageP p) {
	nullPointerException(p);
	return;
}

void crossbowImageMultiply (crossbowImageP p, float value) {
	int i;
	int elements;
	nullPointerException(p);
	invalidConditionException(p->isfloat);
	elements = crossbowImageCurrentElements (p);
	for (i = 0; i < elements; ++i)
		p->data [i] = p->data [i] * value;
	return;
}

void crossbowImageSubtract (crossbowImageP p, float value) {
	int i;
	int elements;
	nullPointerException(p);
	invalidConditionException(p->isfloat);
	elements = crossbowImageCurrentElements (p);
	for (i = 0; i < elements; ++i)
		p->data [i] -= value;
	return;
}

void crossbowImageCheckBounds (crossbowImageP p, float lower, float upper) {
	int i;
	int elements;
	int errors;
	nullPointerException(p);
	invalidConditionException(p->isfloat);
	elements = crossbowImageCurrentElements (p);
	errors = 0;
	for (i = 0; i < elements; ++i) {
		if ((p->data [i] < lower) && (p->data [i] > upper))
			errors ++;
	}
	if (errors) {
		err("%d/%d image pixels out of bounds\n", errors, elements);
	}
	return;
}

char *crossbowImageString (crossbowImageP p) {
	nullPointerException(p);
	if (! p->decoded)
		return NULL;
	char s [1024];
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	crossbowStringAppend (s, &offset, &remaining, "input (%4d x %4d x %d) output (%d x %4d x %4d)",
			/* Input format */
			crossbowImageInputHeight (p), crossbowImageInputWidth (p), crossbowImageChannels (p),
			/* Output format */
			crossbowImageChannels (p), p->height_, p->width_);

	return crossbowStringCopy (s);
}

char *crossbowImageInfo (crossbowImageP p) {
	nullPointerException(p);
	invalidConditionException (p->decoded);
	invalidConditionException (p->isfloat);
	char s [256];
	int offset;
	size_t remaining;
	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;
	crossbowStringAppend (s, &offset, &remaining, "%d x %d x %d", p->height, p->width, p->channels);
	return crossbowStringCopy (s);
}

double crossbowImageChecksum (crossbowImageP p) {
	int i;
	double checksum = 0.0D;

	nullPointerException(p);
	invalidConditionException (p->decoded);
	invalidConditionException (p->isfloat);

	int elements = crossbowImageCurrentElements (p);
	for (i = 0; i < elements; ++i)
		checksum += (double) p->data[i];
	return checksum;
}

void crossbowImageDump (crossbowImageP p, int pixels) {
	int i, j;
	nullPointerException(p);
	invalidConditionException (p->decoded);

	char *info = crossbowImageInfo(p);
	fprintf(stdout, "=== [image: %s, checksum %.5f] ===\n", info, crossbowImageChecksum(p));
	crossbowStringFree(info);

	/* Print the first few `pixels` (x 3) */
	for (i = 0; i < pixels; ++i) {
		for (j = 0; j < 3; ++j)
			fprintf (stdout, "%3u ", p->img[i * 3 + j]);
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "...\n");

	fprintf(stdout, "=== [End of image dump] ===\n");
	fflush(stdout);
}

void crossbowImageDumpAsFloat (crossbowImageP p, int pixels) {
	int i, j;
	nullPointerException(p);
	invalidConditionException (p->decoded);

	char *info = crossbowImageInfo(p);
	fprintf(stdout, "=== [image: %s, checksum %.5f] ===\n", info, crossbowImageChecksum(p));
	crossbowStringFree(info);

	/* Print the first few `pixels` (x 3) */
	for (i = 0; i < pixels; ++i) {
		for (j = 0; j < 3; ++j)
			fprintf (stdout, "%13.8f ", p->data[i * 3 + j]);
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "...\n");

	fprintf(stdout, "=== [End of image dump] ===\n");
	fflush(stdout);
}


int crossbowImageCopy (crossbowImageP p, void * buffer, int offset, int limit) {
    
    nullPointerException (p);
	
    invalidConditionException (p->decoded);
    invalidConditionException (p->isfloat);
    
    int length = crossbowImageCurrentElements (p) * sizeof(float);
    
    if (limit > 0)
        invalidConditionException ((offset + length) < (limit + 1));
    
    /* Copy p->data to buffer (starting at offset) */
    memcpy ((void *)(buffer + offset), (void *)(p->data), length);
	
    return length;
}

void crossbowImageFree (crossbowImageP p) {
	if (! p)
		return;
	if (p->img)
		crossbowFree (p->img, p->elements);
	if (p->data)
		crossbowFree (p->data, crossbowImageCurrentElements (p) * sizeof(float));
	/* Release JPEG library state */
	jpeg_destroy_decompress(p->info);
	crossbowFree (p->info, sizeof(struct jpeg_decompress_struct));
	crossbowFree (p->err,  sizeof(struct jpeg_error_mgr));
	crossbowFree (p, sizeof(crossbow_image_t));
	return;
}

/* Assuming M inputs & N outputs */
static inline void crossbowComputeInterpolationWeights
	(crossbowInterpolationWeightP *weights, long N, long M, float scale) {

	float in;
	long idx;

	weights [N]->lower = 0;
	weights [N]->upper = 0;

	for (idx = N - 1; idx >= 0; --idx) {

		in = idx * scale;

		weights [idx]->lower = (long) in;
		weights [idx]->upper = min(weights[idx]->lower + 1, M - 1);
		weights [idx]->lerp = in - weights[idx]->lower;
	}
}

static inline float crossbowComputeLerp (float topleft, float topright, float bottomleft, float bottomright, float xlerp, float ylerp) {

	float top = topleft + (topright - topleft) * xlerp;
	float bottom = bottomleft + (bottomright - bottomleft) * xlerp;

	return (top + (bottom - top) * ylerp);
}

static void crossbowResizeImage (
	float *input,
	long C,
	long H, /* Input dimensions */
	long W,
	long H_, /* Output dimensions */
	long W_,
	crossbowInterpolationWeightP *X,
	crossbowInterpolationWeightP *Y,
	float *output) {

	long x, y;

	float topleft [3], topright [3], bottomleft [3], bottomright [3];

	/* Resize image from (C, H, W) to (C, H_, W_)... */

	long row  = W  * C;
	long row_ = W_ * C;

	(void) H;

	float *outputP = output;

	for (y = 0; y < H_; ++y) {

		float *_y_lower = input + Y [y]->lower * row;
		float *_y_upper = input + Y [y]->upper * row;

		float  _y_lerp  = Y[y]->lerp;

		for (x = 0; x < W_; ++x) {

			long  _x_lower = X [x]->lower;
			long  _x_upper = X [x]->upper;
			float _x_lerp  = X [x]->lerp;

			/* Read channel #0 (R) */
			topleft     [0] = _y_lower [_x_lower + 0];
			topright    [0] = _y_lower [_x_upper + 0];
			bottomleft  [0] = _y_upper [_x_lower + 0];
			bottomright [0] = _y_upper [_x_upper + 0];

			/* Read channel #1 (G) */
			topleft     [1] = _y_lower [_x_lower + 1];
			topright    [1] = _y_lower [_x_upper + 1];
			bottomleft  [1] = _y_upper [_x_lower + 1];
			bottomright [1] = _y_upper [_x_upper + 1];

			/* Read channel #2 (B) */
			topleft     [2] = _y_lower [_x_lower + 2];
			topright    [2] = _y_lower [_x_upper + 2];
			bottomleft  [2] = _y_upper [_x_lower + 2];
			bottomright [2] = _y_upper [_x_upper + 2];

			/* Compute output */
			outputP [x * C + 0] = crossbowComputeLerp (topleft [0], topright [0], bottomleft [0], bottomright [0], _x_lerp, _y_lerp);
			outputP [x * C + 1] = crossbowComputeLerp (topleft [1], topright [1], bottomleft [1], bottomright [1], _x_lerp, _y_lerp);
			outputP [x * C + 2] = crossbowComputeLerp (topleft [2], topright [2], bottomleft [2], bottomright [2], _x_lerp, _y_lerp);
		}

		/* Move output pointer to next row */
		outputP = outputP + row_;
	}
	return;
}

static inline float crossbowCalculateResizeScale (int inputSize, int outputSize) {
	return (float) inputSize / (float) outputSize;
}

void crossbowImageResize (crossbowImageP p, int height, int width) {
	int i;
	nullPointerException(p);

	/* Allocate interpolation weights on the X and Y dimensions */

	crossbowInterpolationWeightP *X =
			(crossbowInterpolationWeightP *) crossbowMalloc ((width  + 1) * sizeof(crossbowInterpolationWeightP));

	crossbowInterpolationWeightP *Y =
			(crossbowInterpolationWeightP *) crossbowMalloc ((height + 1) * sizeof(crossbowInterpolationWeightP));

	for (i = 0; i < width + 1; ++i)
		X[i] = (crossbowInterpolationWeightP) crossbowMalloc (sizeof(crossbow_interpolation_weight_t));

	for (i = 0; i < height + 1; ++i)
		Y[i] = (crossbowInterpolationWeightP) crossbowMalloc (sizeof(crossbow_interpolation_weight_t));

	crossbowComputeInterpolationWeights (X,  width, crossbowImageCurrentWidth  (p), crossbowCalculateResizeScale (crossbowImageCurrentWidth  (p),  width));
	crossbowComputeInterpolationWeights (Y, height, crossbowImageCurrentHeight (p), crossbowCalculateResizeScale (crossbowImageCurrentHeight (p), height));

	for (i = 0; i < width + 1; ++i) {
		X[i]->lower *= p->channels;
		X[i]->upper *= p->channels;
	}

	/* Allocate new buffer */
	float *resized = (float *) crossbowMalloc (crossbowImageChannels (p) * height * width * sizeof(float));

	crossbowResizeImage (
		p->data,
		p->channels,
		crossbowImageCurrentHeight (p),
		crossbowImageCurrentWidth  (p),
		height,
		width,
		X,
		Y,
		resized
	);

	/* Free interpolation weights */

	for (i = 0; i < width + 1; ++i)
		crossbowFree (X[i], sizeof(crossbow_interpolation_weight_t));

	for (i = 0; i < height + 1; ++i)
		crossbowFree (Y[i], sizeof(crossbow_interpolation_weight_t));

	crossbowFree (X, ((width  + 1) * sizeof(crossbowInterpolationWeightP)));
	crossbowFree (Y, ((height + 1) * sizeof(crossbowInterpolationWeightP)));

	/* Free current data pointer */
	crossbowFree (p->data, crossbowImageCurrentElements (p) * sizeof(float));
	/* Assign new data pointer */
	p->data = resized;
	/* Set current data, height & width */
	p->height = height;
	p->width  =  width;

	return;
}

static unsigned crossbowImageGenerateRandomCrop (crossbowRectangleP p, int width, int height, float *area, float ratio) {
	
	nullPointerException (p);
	
	/* If any of the width, height, min/max area, or aspect ratio is less or equal to 0, return false */
	invalidConditionException(width   > 0);
	invalidConditionException(height  > 0);
	invalidConditionException(ratio   > 0);
	invalidConditionException(area[0] > 0);
	invalidConditionException(area[1] > 0);
	invalidConditionException(area[0] <= area[1]);
	
	/* Compute min and max relative crop area */
	float minArea = area[0] * width * height;
	float maxArea = area[1] * width * height;

	int minHeight = (int) lrintf (sqrt (minArea / ratio));
	int maxHeight = (int) lrintf (sqrt (maxArea / ratio));
	
	/* Find smaller max height s.t. round (maxHeight x ratio) <= width */
	if (lrintf (maxHeight * ratio) > width) {
		float epsilon = 0.0000001;
		maxHeight = (int) ((width + 0.5 - epsilon) / ratio);
	}
	
	if (maxHeight > height)
		maxHeight = height;
	
	if (minHeight >= maxHeight)
		minHeight = maxHeight;
	
	if (minHeight < maxHeight) {
		/* Generate a random number of the closed range [0, (maxHeight - minHeight)] */
		/* minHeight += crossbowYarngNext (0, maxHeight - minHeight + 1); */
		minHeight += crossbowYarngNext (0, maxHeight - minHeight);
	}
	
	int minWidth = (int) lrintf (minHeight * ratio);
	/* Check that width is less or equal to the original width */
	invalidConditionException(minWidth <= width);
	
	float newArea = (float) (minHeight * minWidth);

	/* Deal with rounding errors */
	if (newArea < minArea) {
		/* Try a bigger rectangle */
		minHeight += 1;
		minWidth = (int) lrintf (minHeight * ratio);
		newArea = (float) (minHeight * minWidth);
	}
	
	if ((newArea < minArea) || (newArea > maxArea) || (minWidth <= 0) || (minWidth > width) || (minHeight <= 0) || (minHeight > height)) {
		return 0;
	}
	
	int y = 0;
	if (minHeight < height)
		y = crossbowYarngNext (0, height - minHeight);
	
	int x = 0;
	if (minWidth < width)
		x = crossbowYarngNext (0, width - minWidth);
	
	/* Configure rectangle */
	crossbowRectangleSet (p, x, y, x + minWidth, y + minHeight);

	return 1;
}

void crossbowImageSampleDistortedBoundingBox (crossbowImageP p, crossbowArrayListP boxes, float coverage, float *ratio, float *area, int attempts, int *height, int *width, int *top, int *left) {
	int idx;
	nullPointerException(p);

	int h = (int) crossbowImageCurrentHeight (p);
	int w = (int) crossbowImageCurrentWidth  (p);
	
	/* Convert bounding boxes to rectangles. If there are none, use entire image by default */
	dbg("Convert bounding boxes to rectangles\n");
	crossbowArrayListP rectangles = NULL;
	if (boxes) {
		/* Allocate as many slots as the number of boxes */
		rectangles = crossbowArrayListCreate (crossbowArrayListSize (boxes));
		for (idx = 0; idx < crossbowArrayListSize (boxes); ++idx) {

			crossbowBoundingBoxP box = (crossbowBoundingBoxP) crossbowArrayListGet (boxes, idx);
			if (! crossbowBoundingBoxIsValid(box))
				err("Invalid bounding box");
			
			int xmin = box->xmin * w;
			int ymin = box->ymin * h;
			int xmax = box->xmax * w;
			int ymax = box->ymax * h;
			
			crossbowRectangleP rectangle = crossbowRectangleCreate (xmin, ymin, xmax, ymax);
			crossbowArrayListSet (rectangles, idx, rectangle);
		}
	} else {
		rectangles = crossbowArrayListCreate (1);
		crossbowArrayListSet (rectangles, 0, crossbowRectangleCreate (0, 0, w, h));
	}
	
	crossbowRectangleP crop = crossbowRectangleCreate (0, 0, 0, 0);
	unsigned generated = 0;
	int i;
	for (i = 0; i < attempts; ++i) {
		
		/* Sample aspect ratio (within ratio bounds) */
		float sample = crossbowYarngNext (0, 1) * (ratio[1] - ratio[0]) + ratio[0];
		
		dbg("Generate random crop\n");
		if (crossbowImageGenerateRandomCrop (crop, w, h, area, sample)) {

			dbg("Check coverage\n");
			if (crossbowRectangleCovers(crop, coverage, rectangles)) {
				generated = 1;
				break;
			}
		}
	}
	if (! generated) {
		/* Set the entire image as the bounding box */
		crossbowRectangleSet (crop, 0, 0, w, h);
	}

	/* Determine cropping parameters for the bounding box */
	dbg("Set cropping parameters\n");
	*width  = crop->xmax - crop->xmin;
	*height = crop->ymax - crop->ymin;
	
	/* be careful of the order */
	*top  = crop->ymin;
	*left = crop->xmin;

	/* Ensure sampled bounding box fits current image dimensions */
	invalidConditionException (w >= (*left + *width));
	invalidConditionException (h >= (*top  + *height));
	
	/* Free local state */
	
	for (idx = 0; idx < crossbowArrayListSize (rectangles); ++idx) {
		crossbowRectangleP rect = (crossbowRectangleP) crossbowArrayListGet (rectangles, idx);
		crossbowRectangleFree (rect);
	}
	crossbowArrayListFree (rectangles);
	
	crossbowRectangleFree (crop);
	
	return;
}

