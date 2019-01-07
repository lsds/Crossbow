#include "cudnnconvparams.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#include "cudnnhelper.h"

crossbowCudnnConvParamsP crossbowCudnnConvParamsCreate () {

	crossbowCudnnConvParamsP p;

	p = (crossbowCudnnConvParamsP) crossbowMalloc (sizeof(crossbow_cudnn_conv_params_t));
	
	/* Initialise to 0 */
	memset (p, 0, sizeof(crossbow_cudnn_conv_params_t));

	return p;
}

void crossbowCudnnConvParamsSetInputDescriptor (crossbowCudnnConvParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	p->input = crossbowCudnnTensorCreate (count, channels, height, width);
	return;
}

void crossbowCudnnConvParamsSetOutputDescriptor (crossbowCudnnConvParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	p->output = crossbowCudnnTensorCreate (count, channels, height, width);
	return;
}

void crossbowCudnnConvParamsSetConvolutionDescriptor (crossbowCudnnConvParamsP p, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth) {
	nullPointerException(p);
	checkCudnnStatus(cudnnCreateConvolutionDescriptor(&(p->convDesc)));
#if CUDNN_MAJOR >= 6
	checkCudnnStatus(cudnnSetConvolution2dDescriptor(p->convDesc, paddingHeight, paddingWidth, strideHeight, strideWidth, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
	checkCudnnStatus(cudnnSetConvolution2dDescriptor(p->convDesc, paddingHeight, paddingWidth, strideHeight, strideWidth, 1, 1, CUDNN_CROSS_CORRELATION));
#endif
	return;
}

void crossbowCudnnConvParamsSetFilterDescriptor (crossbowCudnnConvParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	dbg("Set convolution filter descriptor [%d, %d, %d, %d]\n", count, channels, height, width);
	checkCudnnStatus(cudnnCreateFilterDescriptor(&(p->filterDesc)));
	checkCudnnStatus(cudnnSetFilter4dDescriptor(p->filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, count, channels, height, width));
	return;
}

void crossbowCudnnConvParamsSetBiasDescriptor (crossbowCudnnConvParamsP p, int count, int channels, int height, int width) {
	nullPointerException(p);
	dbg("Set convolution bias descriptor [%d, %d, %d, %d]\n", count, channels, height, width);
	checkCudnnStatus(cudnnCreateTensorDescriptor(&(p->biasDesc)));
	checkCudnnStatus(cudnnSetTensor4dDescriptor(p->biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, channels, height, width));
	p->hasBias = 1;
	return;
}

static cudnnConvolutionFwdAlgo_t crossbowCudnnConvParamsAnalyseForwardAlgorithmPerf (crossbowCudnnConvParamsP p, double threshold, cudnnHandle_t handle) {

	cudnnConvolutionFwdAlgo_t algo;
	int ndx;
	int returnedAlgoCount, requestedAlgoCount = 8; /* There are 8 algorithms in total */
	cudnnConvolutionFwdAlgoPerf_t *perfResults = NULL;

	/* Allocate performance results */
	perfResults = (cudnnConvolutionFwdAlgoPerf_t *) crossbowMalloc (requestedAlgoCount * sizeof(cudnnConvolutionFwdAlgoPerf_t));

	checkCudnnStatus(cudnnFindConvolutionForwardAlgorithm (
			handle,
			p->input->descriptor,
			p->filterDesc,
			p->convDesc,
			p->output->descriptor,
			requestedAlgoCount,
			&returnedAlgoCount,
			perfResults
			));

#ifdef GPU_VERBOSE
	info("%d/%d forward algorithms returned:\n", returnedAlgoCount, requestedAlgoCount);
	for (ndx = 0; ndx < returnedAlgoCount; ++ndx) {

		info("%2d: %-60s time %10.5f ms memory %10zu bytes %s\n",
			ndx,
			cudnnConvolutionFwdAlgorithmString ( perfResults[ndx].algo),
			perfResults[ndx].time,
			perfResults[ndx].memory,
			cudnnGetErrorString(perfResults[ndx].status)
			);
	}
#endif
	
	/* Trade-off memory for performance */
	int alt = 0;
	if (threshold > 0) {
		double slowdown = 0.0;
		for (ndx = 1; ndx < returnedAlgoCount; ++ndx) {
			if (perfResults[ndx].time > 0) {
				slowdown = perfResults[ndx].time / perfResults[0].time;
				if (perfResults[ndx].memory < perfResults[0].memory && slowdown < threshold)
					alt = ndx;
			}
		}
	}

	/* Return fastest algorithm */
	algo = perfResults[alt].algo;

	/* Free performance results */
	crossbowFree (perfResults, requestedAlgoCount * sizeof(cudnnConvolutionFwdAlgoPerf_t));

	return algo;
}

static size_t cudnnConvolutionFwdPreferenceMemoryLimit (cudnnConvolutionFwdPreference_t preference, int memoryLimitInBytes) {

	size_t limit = 0;

	dbg("Forward algorithm preference is %s\n", cudnnConvolutionFwdPreferenceString (preference));

	if (preference == CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
		limit = (size_t) memoryLimitInBytes;

	dbg("User memory limit set to %zu\n", limit);

	return limit;
}

size_t crossbowCudnnConvParamsConfigureForward (crossbowCudnnConvParamsP p, int memoryLimitInBytes, double threshold, cudnnHandle_t handle) {

	cudnnConvolutionFwdPreference_t preference;
	size_t limit = 0;

	if (memoryLimitInBytes == 0)
	{
		preference = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
	}
	else if (memoryLimitInBytes < 0)
	{
		preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	}
	else
	{
		preference = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
	}

	limit = cudnnConvolutionFwdPreferenceMemoryLimit (preference, memoryLimitInBytes);

	if (preference != CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) {

		checkCudnnStatus(cudnnGetConvolutionForwardAlgorithm (
			handle,
			p->input->descriptor,
			p->filterDesc,
			p->convDesc,
			p->output->descriptor,
			preference,
			limit,
			&(p->forwardAlgorithm)
			));
	}
	else {

		p->forwardAlgorithm = crossbowCudnnConvParamsAnalyseForwardAlgorithmPerf (p, threshold, handle);
	}

	checkCudnnStatus(cudnnGetConvolutionForwardWorkspaceSize (
			handle,
			p->input->descriptor,
			p->filterDesc,
			p->convDesc,
			p->output->descriptor,
			p->forwardAlgorithm,
			&(p->forwardWorkSpaceSize)
			));

	dbg("Chosen forward algorithm is %s (%zu bytes required)\n", cudnnConvolutionFwdAlgorithmString (p->forwardAlgorithm), p->forwardWorkSpaceSize);

	return p->forwardWorkSpaceSize;
}

static cudnnConvolutionBwdDataAlgo_t crossbowCudnnConvParamsAnalyseBackwardDataAlgorithmPerf (crossbowCudnnConvParamsP p, double threshold, cudnnHandle_t handle) {

	cudnnConvolutionBwdDataAlgo_t algo;
	int ndx;
	int returnedAlgoCount, requestedAlgoCount = 6; /* There are 8 algorithms in total */
	cudnnConvolutionBwdDataAlgoPerf_t *perfResults = NULL;

	/* Allocate performance results */
	perfResults = (cudnnConvolutionBwdDataAlgoPerf_t *) crossbowMalloc (requestedAlgoCount * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));

	checkCudnnStatus(cudnnFindConvolutionBackwardDataAlgorithm (
			handle,
			p->filterDesc,
			p->output->descriptor,
			p->convDesc,
			p->input->descriptor,
			requestedAlgoCount,
			&returnedAlgoCount,
			perfResults
	));

#ifdef GPU_VERBOSE
	info("%d/%d backward data algorithms returned:\n", returnedAlgoCount, requestedAlgoCount);
	for (ndx = 0; ndx < returnedAlgoCount; ++ndx) {

		info("%2d: %-60s time %10.5f ms memory %10zu bytes %s\n",
				ndx,
				cudnnConvolutionBwdDataAlgorithmString ( perfResults[ndx].algo),
				perfResults[ndx].time,
				perfResults[ndx].memory,
				cudnnGetErrorString(perfResults[ndx].status)
		);
	}
#endif
	
	/* Trade-off memory for performance */
	int alt = 0;
	if (threshold > 0) {
		double slowdown = 0.0;
		for (ndx = 1; ndx < returnedAlgoCount; ++ndx) {
			if (perfResults[ndx].time > 0) {
				slowdown = perfResults[ndx].time / perfResults[0].time;
				if (perfResults[ndx].memory < perfResults[0].memory && slowdown < threshold)
					alt = ndx;
			}
		}
	}

	algo = perfResults[alt].algo;

	/* Free performance results */
	crossbowFree (perfResults, requestedAlgoCount * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));

	return algo;
}

static size_t cudnnConvolutionBwdDataPreferenceMemoryLimit (cudnnConvolutionBwdDataPreference_t preference, int memoryLimitInBytes) {

	size_t limit = 0;

	dbg("Backward data algorithm preference is %s\n", cudnnConvolutionBwdDataPreferenceString (preference));

	if (preference == CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
		limit = (size_t) memoryLimitInBytes;

	dbg("User memory limit set to %zu\n", limit);

	return limit;
}

size_t crossbowCudnnConvParamsConfigureBackwardData (crossbowCudnnConvParamsP p, int memoryLimitInBytes, double threshold, cudnnHandle_t handle) {

	cudnnConvolutionBwdDataPreference_t preference;
	size_t limit = 0;

	if (memoryLimitInBytes == 0)
	{
		preference = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
	}
	else if (memoryLimitInBytes < 0)
	{
		preference = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
	}
	else
	{
		preference = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
	}

	limit = cudnnConvolutionBwdDataPreferenceMemoryLimit (preference, memoryLimitInBytes);

	if (preference != CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST) {

		checkCudnnStatus(cudnnGetConvolutionBackwardDataAlgorithm (
			handle,
			p->filterDesc,
			p->output->descriptor,
			p->convDesc,
			p->input->descriptor,
			preference,
			limit,
			&(p->backwardDataAlgorithm)
			));
	}
	else {

		p->backwardDataAlgorithm = crossbowCudnnConvParamsAnalyseBackwardDataAlgorithmPerf (p, threshold, handle);
	}
#ifdef REPRODUCIBILITY
	/* Override chosen configuration in favor of reproducible results */
	p->backwardDataAlgorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
#endif
	checkCudnnStatus(cudnnGetConvolutionBackwardDataWorkspaceSize (
			handle,
			p->filterDesc,
			p->output->descriptor,
			p->convDesc,
			p->input->descriptor,
			p->backwardDataAlgorithm,
			&(p->backwardDataWorkSpaceSize)
			));

	dbg("Chosen backward data algorithm is %s (%zu bytes required)\n", cudnnConvolutionBwdDataAlgorithmString (p->backwardDataAlgorithm), p->backwardDataWorkSpaceSize);

	return p->backwardDataWorkSpaceSize;
}

static cudnnConvolutionBwdFilterAlgo_t crossbowCudnnConvParamsAnalyseBackwardFilterAlgorithmPerf (crossbowCudnnConvParamsP p, double threshold, cudnnHandle_t handle) {

	cudnnConvolutionBwdFilterAlgo_t algo;
	int ndx;
	int returnedAlgoCount, requestedAlgoCount = 6; /* There are 8 algorithms in total */
	cudnnConvolutionBwdFilterAlgoPerf_t *perfResults = NULL;

	/* Allocate performance results */
	perfResults = (cudnnConvolutionBwdFilterAlgoPerf_t *) crossbowMalloc (requestedAlgoCount * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));

	checkCudnnStatus(cudnnFindConvolutionBackwardFilterAlgorithm (
			handle,
			p->input->descriptor,
			p->output->descriptor,
			p->convDesc,
			p->filterDesc,
			requestedAlgoCount,
			&returnedAlgoCount,
			perfResults
	));

#ifdef GPU_VERBOSE
	info("%d/%d backward filter algorithms returned:\n", returnedAlgoCount, requestedAlgoCount);
	for (ndx = 0; ndx < returnedAlgoCount; ++ndx) {

		info("%2d: %-60s time %10.5f ms memory %10zu bytes %s\n",
				ndx,
				cudnnConvolutionBwdFilterAlgorithmString ( perfResults[ndx].algo),
				perfResults[ndx].time,
				perfResults[ndx].memory,
				cudnnGetErrorString(perfResults[ndx].status)
		);
	}
#endif
	
	/* Trade-off memory for performance */
	int alt = 0;
	if (threshold > 0) {
		double slowdown = 0.0;
		for (ndx = 1; ndx < returnedAlgoCount; ++ndx) {
			if (perfResults[ndx].time > 0) {
				slowdown = perfResults[ndx].time / perfResults[0].time;
				if (perfResults[ndx].memory < perfResults[0].memory && slowdown < threshold)
					alt = ndx;
			}
		}
	}

	algo = perfResults[alt].algo;

	/* Free performance results */
	crossbowFree (perfResults, requestedAlgoCount * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));

	return algo;
}

static size_t cudnnConvolutionBwdFilterPreferenceMemoryLimit (cudnnConvolutionBwdFilterPreference_t preference, int memoryLimitInBytes) {

	size_t limit = 0;

	dbg("Backward filter algorithm preference is %s\n", cudnnConvolutionBwdFilterPreferenceString (preference));

	if (preference == CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
		limit = (size_t) memoryLimitInBytes;

	dbg("User memory limit set to %zu\n", limit);

	return limit;
}

size_t crossbowCudnnConvParamsConfigureBackwardFilter (crossbowCudnnConvParamsP p, int memoryLimitInBytes, double threshold, cudnnHandle_t handle) {

	cudnnConvolutionBwdFilterPreference_t preference;
	size_t limit = 0;

	if (memoryLimitInBytes == 0)
	{
		preference = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
	}
	else if (memoryLimitInBytes < 0)
	{
		preference = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
	}
	else
	{
		preference = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
	}

	limit = cudnnConvolutionBwdFilterPreferenceMemoryLimit (preference, memoryLimitInBytes);

	if (preference != CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST) {

		checkCudnnStatus(cudnnGetConvolutionBackwardFilterAlgorithm (
			handle,
			p->input->descriptor,
			p->output->descriptor,
			p->convDesc,
			p->filterDesc,
			preference,
			limit,
			&(p->backwardFilterAlgorithm)
			));
	}
	else {

		p->backwardFilterAlgorithm = crossbowCudnnConvParamsAnalyseBackwardFilterAlgorithmPerf (p, threshold, handle);
	}
#ifdef REPRODUCIBILITY
	/* Override chosen configuration in favor of reproducible results */
	p->backwardFilterAlgorithm = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
#endif
	checkCudnnStatus(cudnnGetConvolutionBackwardFilterWorkspaceSize (
			handle,
			p->input->descriptor,
			p->output->descriptor,
			p->convDesc,
			p->filterDesc,
			p->backwardFilterAlgorithm,
			&(p->backwardFilterWorkSpaceSize)
			));

	dbg("Chosen backward filter algorithm is %s (%zu bytes required)\n", cudnnConvolutionBwdFilterAlgorithmString (p->backwardFilterAlgorithm), p->backwardFilterWorkSpaceSize);

	return p->backwardFilterWorkSpaceSize;
}

char *crossbowCudnnConvParamsString (crossbowCudnnConvParamsP p) {

	char s [1024];
	int offset;
	size_t remaining;

	memset (s, 0, sizeof(s));
	remaining = sizeof(s) - 1;
	offset = 0;

	/* Get tensor descriptors */
	char *d1 = crossbowCudnnTensorString (p->input);
	char *d2 = crossbowCudnnTensorString (p->output);

	/* input [n, c, h, w] output [n, c, h, w] */
	crossbowStringAppend (s, &offset, &remaining, "input %s output %s", d1, d2);

	crossbowStringFree (d1);
	crossbowStringFree (d2);

	return crossbowStringCopy (s);
}

void crossbowCudnnConvParamsFree (crossbowCudnnConvParamsP p) {

	if (! p)
		return;

	crossbowCudnnTensorFree (p->input);
	crossbowCudnnTensorFree (p->output);

	/* Free convolution descriptor */
	checkCudnnStatus(cudnnDestroyConvolutionDescriptor(p->convDesc));

	/* Free filter descriptor */
	checkCudnnStatus(cudnnDestroyFilterDescriptor(p->filterDesc));

	/* Free bias descriptor */
	if (p->hasBias)
		checkCudnnStatus(cudnnDestroyTensorDescriptor(p->biasDesc));

	crossbowFree (p, sizeof(crossbow_cudnn_conv_params_t));
}
