#ifndef __CROSSBOW_CUDNN_HELPER_H_
#define __CROSSBOW_CUDNN_HELPER_H_

#include <cudnn.h>

const char *cudnnActivationModeString (cudnnActivationMode_t);

const char *cudnnNanPropagationString (cudnnNanPropagation_t);

const char *cudnnPoolingModeString (cudnnPoolingMode_t);

const char *cudnnConvolutionFwdAlgorithmString (cudnnConvolutionFwdAlgo_t);

const char *cudnnConvolutionBwdFilterAlgorithmString (cudnnConvolutionBwdFilterAlgo_t);

const char *cudnnConvolutionBwdDataAlgorithmString (cudnnConvolutionBwdDataAlgo_t);

const char *cudnnConvolutionFwdPreferenceString (cudnnConvolutionFwdPreference_t);

const char *cudnnConvolutionBwdDataPreferenceString (cudnnConvolutionBwdDataPreference_t);

const char *cudnnConvolutionBwdFilterPreferenceString (cudnnConvolutionBwdFilterPreference_t);

#endif /* __CROSSBOW_CUDNN_HELPER_H_ */
