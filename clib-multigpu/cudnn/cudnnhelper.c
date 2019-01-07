#include "cudnnhelper.h"

const char *cudnnActivationModeString (cudnnActivationMode_t mode) {

	switch (mode) {

	case CUDNN_ACTIVATION_SIGMOID:      return "CUDNN_ACTIVATION_SIGMOID";
	case CUDNN_ACTIVATION_RELU:         return "CUDNN_ACTIVATION_RELU";
	case CUDNN_ACTIVATION_TANH:         return "CUDNN_ACTIVATION_TANH";
	case CUDNN_ACTIVATION_CLIPPED_RELU: return "CUDNN_ACTIVATION_CLIPPED_RELU";
#if CUDNN_MAJOR >= 6
	case CUDNN_ACTIVATION_ELU:          return "CUDNN_ACTIVATION_ELU";
#endif
	}

	return NULL;
}

const char *cudnnNanPropagationString (cudnnNanPropagation_t opt) {

	switch (opt) {

	case CUDNN_NOT_PROPAGATE_NAN: return "CUDNN_NOT_PROPAGATE_NAN";
	case CUDNN_PROPAGATE_NAN:     return "CUDNN_PROPAGATE_NAN";
	}

	return NULL;
}

const char *cudnnPoolingModeString (cudnnPoolingMode_t mode) {

	switch (mode) {

	case CUDNN_POOLING_MAX:                           return "CUDNN_POOLING_MAX";
	case CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING: return "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING";
	case CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING: return "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING";
#if CUDNN_MAJOR >= 6
	case CUDNN_POOLING_MAX_DETERMINISTIC:             return "CUDNN_POOLING_MAX_DETERMINISTIC";
#endif
	}

	return NULL;
}

const char *cudnnConvolutionFwdAlgorithmString (cudnnConvolutionFwdAlgo_t algorithm) {

	switch (algorithm) {

	case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:         return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
	case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
	case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:                  return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
	case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:                return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
	case CUDNN_CONVOLUTION_FWD_ALGO_FFT:                   return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
	case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:            return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
	case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:              return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
	case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:     return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
	default:
		return "Unkwown forward algorithm";
	}

	return NULL;
}

const char *cudnnConvolutionBwdFilterAlgorithmString (cudnnConvolutionBwdFilterAlgo_t algorithm) {

	switch (algorithm) {

	case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:                 return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0";
	case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:                 return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1";
	case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:               return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT";
	case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:                 return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3";
	case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED";
	default:
		return "Unkwown backward filter algorithm";
	}

	return NULL;
}

const char *cudnnConvolutionBwdDataAlgorithmString (cudnnConvolutionBwdDataAlgo_t algorithm) {

	switch (algorithm) {

	case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:                 return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
	case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:                 return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
	case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:               return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
	case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:        return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
	case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:          return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
	case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
	default:
		return "Unkwown backward data algorithm";
	}

	return NULL;
}

const char *cudnnConvolutionFwdPreferenceString (cudnnConvolutionFwdPreference_t preference) {

	switch (preference) {

	case CUDNN_CONVOLUTION_FWD_NO_WORKSPACE:            return "CUDNN_CONVOLUTION_FWD_NO_WORKSPACE";
	case CUDNN_CONVOLUTION_FWD_PREFER_FASTEST:          return "CUDNN_CONVOLUTION_FWD_PREFER_FASTEST";
	case CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT: return "CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT";
	default:
		return "Unknown forward algorithm preference";
	}

	return NULL;
}

const char *cudnnConvolutionBwdDataPreferenceString (cudnnConvolutionBwdDataPreference_t preference) {

	switch (preference) {

	case CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE:            return "CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE";
	case CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST:          return "CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST";
	case CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT: return "CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT";
	default:
		return "Unknown backward data algorithm preference";
	}

	return NULL;
}

const char *cudnnConvolutionBwdFilterPreferenceString (cudnnConvolutionBwdFilterPreference_t preference) {

	switch (preference) {

	case CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE:            return "CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE";
	case CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:          return "CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST";
	case CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT: return "CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT";
	default:
		return "Unknown backward filter algorithm preference";
	}

	return NULL;
}
