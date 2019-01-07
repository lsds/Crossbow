#include "datatransform.h"

__global__ void crossbowKernelDataTransformKernel (
	const int    count,
	const int    channels,
	const int    inputImageHeight,
	const int    inputImageWidth,
	const int    outputImageHeight,
	const int    outputImageWidth,
	const int    cropSize,
	const bool   mirror,
	const bool   training,
	const float  scaleFactor,
	const bool   substractMean,
	const bool   hasMeanImage,
	const float* input,
	float*       output,
	const float* means,
	const unsigned int* randoms) {
	
	const int c = blockIdx.y;
	
	const int  inputImageDimensions2D = inputImageHeight * inputImageWidth;
	const int  inputImageDimensions3D = inputImageHeight * inputImageWidth * channels;

	const int outputImageDimensions2D = outputImageHeight * outputImageWidth;
	const int outputImageDimensions3D = outputImageHeight * outputImageWidth * channels;

	/* Loop over images */
	for (int n = blockIdx.x; n < count; n += gridDim.x) {
		
		/* Move pointer in `randoms` */
		const unsigned int *randomP = &randoms[n * 3];
		
		bool _mirror = mirror && (randomP[0] % 2); /* 0 or 1 */
		if (! training)
			_mirror = 0;
		
		int heightOffset = 0;
		int  widthOffset = 0;
		
		if (cropSize > 0) {
			if (training) {
				 heightOffset = randomP[1] % (inputImageHeight - cropSize + 1);
				  widthOffset = randomP[2] % (inputImageWidth  - cropSize + 1);
			} 
			else {
				heightOffset = (inputImageHeight - cropSize) / 2;
			 	 widthOffset = (inputImageWidth  - cropSize) / 2;
			}
		}
		
		const float *inputP = &input[n *  inputImageDimensions3D + c *  inputImageDimensions2D];
		float *outputP = &output[n * outputImageDimensions3D + c * outputImageDimensions2D];

		/* Loop over pixels */
		for (int h = threadIdx.y; h < outputImageHeight; h += blockDim.y) {

			for (int w = threadIdx.x; w < outputImageWidth; w += blockDim.x) {

				int  inputIndex = (heightOffset + h) * inputImageWidth + widthOffset + w;
				int outputIndex = (_mirror) ? (h * outputImageWidth + (outputImageWidth - 1 - w)) : (h * outputImageWidth + w);

				float pixel = inputP[inputIndex];

				/* Transform pixel */
				if (substractMean)
					pixel = (hasMeanImage) ? (pixel - means[c * inputImageDimensions2D + inputIndex]) : (pixel - means[c]);

				outputP[outputIndex] = pixel * scaleFactor;
			}
		}
	}
	
	return;
}

void crossbowKernelDataTransform (void *args) {

	/* Input and output variables */
	crossbowDataBufferP input, output;

	crossbowStreamP s = (crossbowStreamP) args;
	/* checkCublasStatus(cublasSetStream (s->handle, s->stream)); */

	/* Set input buffer (namely, examples) */
	input = crossbowStreamGetCurrentInput (s);

	/* Get an output variable buffer */
	output = crossbowStreamGetCurrentOutput (s);

	/* Get kernel configuration parameters */

	int   cropSize     = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 0));
	int   mirror       = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 1));
	float scaleFactor  = crossbowKernelConfigParamGetFloatValue ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 2));
	int   subtractMean = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 3));
	int   hasMeanImage = crossbowKernelConfigParamGetIntValue   ((crossbowKernelConfigParamP) crossbowArrayListGet(s->op->kernel->parameters, 4));

	/* Get local variables, if any */
	/* Local variables */
	crossbowDataBufferP   means = NULL;
	crossbowDataBufferP randoms = NULL;

	int localvariableid = 0;
	if (subtractMean) {
		means = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, localvariableid++), s->deviceId, s->id, NULL, NULL);
	}
	if (cropSize > 0 || mirror > 0) {
		int elements, length = 0;
		randoms  = crossbowLocalVariableGetDataBuffer ((crossbowLocalVariableP) crossbowArrayListGet (s->op->kernel->variables, localvariableid++), s->deviceId, s->id, NULL, &length);
		elements = length / 4;
		dbg("%d random elements\n", elements);
#ifndef CURAND_NOOP
		checkCurandStatus (curandGenerate (s->curandGenerator, (unsigned int *) (randoms->dev), elements));
#else
        /* Subterfuge unused parameter warnings */
        UNUSED (randoms);
        UNUSED (elements);
#endif
    }

	int examples = crossbowVariableSchemaShape (s->op->kernel->inputs[0]->schema, 0);
	int channels = crossbowVariableSchemaShape (s->op->kernel->inputs[0]->schema, 1);

	int inputImageHeight = crossbowVariableSchemaShape (s->op->kernel->inputs[0]->schema, 2);
	int inputImageWidth  = crossbowVariableSchemaShape (s->op->kernel->inputs[0]->schema, 3);

	dbg("Input shape is %d x %d x %d x %d\n", examples, channels, inputImageHeight, inputImageWidth);

	int outputImageHeight = crossbowVariableSchemaShape (s->op->kernel->output->schema, 2);
	int outputImageWidth  = crossbowVariableSchemaShape (s->op->kernel->output->schema, 3);

	dbg("Output shape is %d x %d x %d x %d\n", examples, channels, outputImageHeight, outputImageWidth);

	/* cudaMemsetAsync (((void *) output->dev), 0, 4 * examples * channels * outputImageHeight * outputImageWidth, s->stream); */

	dim3 grid (examples, channels);
	dim3 block (16, 16);

#ifndef KERNEL_NOOP
	crossbowKernelDataTransformKernel<<<grid, block, 0, s->stream[s->op->branch]>>>(
			examples,
			channels,
			inputImageHeight,
			inputImageWidth,
			outputImageHeight,
			outputImageWidth,
			cropSize,
			(mirror == 1),
			(s->phi == TRAIN),
			scaleFactor,
			subtractMean,
			hasMeanImage,
			(float *)  (input->dev),
            (float *) (output->dev),
			(subtractMean) ? (float *) (means->dev) : NULL,
			(cropSize > 0 || mirror == 1) ? (unsigned int *) (randoms->dev) : NULL
			);
#else
    /* Subterfuge unused parameter warnings */
    UNUSED (grid);
    UNUSED (block);
    UNUSED (channels);
    UNUSED (examples);
    UNUSED (channels);
    UNUSED (inputImageHeight);
    UNUSED (inputImageWidth);
    UNUSED (outputImageHeight);
    UNUSED (outputImageWidth);
    UNUSED (cropSize);
    UNUSED (mirror);
    UNUSED (scaleFactor);
    UNUSED (subtractMean);
    UNUSED (hasMeanImage);
    UNUSED (input);
    UNUSED (output);
    UNUSED (means);
    UNUSED (randoms);
#endif

	/* Store output in stream */
	crossbowListAppend(s->outputs[s->op->id], output);

	/* Return local variables */
	if (means)
		crossbowListAppend(s->locals[s->op->id], means);

	if (randoms)
		crossbowListAppend(s->locals[s->op->id], randoms);

	return;
}
