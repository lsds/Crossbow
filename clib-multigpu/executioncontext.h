#ifndef __CROSSBOW_EXECUTION_CONTEXT_H_
#define __CROSSBOW_EXECUTION_CONTEXT_H_

#include "batch.h"

#include "kernelmap.h"
#include "threadsafequeue.h"
#include "arraylist.h"

#include "model.h"
#include "modelmanager.h"

#include "resulthandler.h"

#include "callbackhandler.h"

#include "lightweightdatasethandler.h"

#include "stream.h"

#include "device.h"

#include "utils.h"

#ifdef INTER_TASK_MEASUREMENTS
#include "timer.h"
#include "measurementlist.h"
#endif

#include "recorddataset.h"

#include <jni.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#include <curand.h>

#include <nvToolsExt.h>

#include <nccl.h>

typedef struct crossbow_execution_context *crossbowExecutionContextP;
typedef struct crossbow_execution_context {
	
	crossbowArrayListP devices;

	int defaultDeviceId;
	crossbowModelSynchronisationMode_t mode;
	
	/* crossbowThreadSafeQueueP streams; */
	crossbowArrayListP streams;
	
	int previousStreamId;

	crossbowKernelMapP kernelmap;
	
	crossbowArrayListP   kernels;
	crossbowArrayListP dataflows;
	
	crossbowModelP theModel;
	crossbowModelManagerP modelmanager;

	crossbowResultHandlerP resulthandler;
	
	crossbowLightWeightDatasetHandlerP datasethandler;

	crossbowBatchP batch;

	crossbowArrayListP callbackhandlers;
#ifdef USE_TASKHANDLERS
	crossbowArrayListP taskhandlers;
#endif

#ifdef INTER_TASK_MEASUREMENTS
	crossbowTimerP timer;
	crossbowMeasurementListP measurements;
#endif

#ifdef USE_NCCL
	int nc; /* Number of comms (i.e. active devices) */
	ncclComm_t *comms;
	int *devs;
#endif

	unsigned long long seed;
	unsigned long long version; /* Checkpoint version */

	crossbowRecordDatasetP dataset[2]; /* 0: training; 1: test dataset */

} crossbow_execution_context_t;

crossbowExecutionContextP crossbowExecutionContextCreate (crossbowArrayListP, int, int, int, int *);

void crossbowExecutionContextBindKernels (crossbowExecutionContextP);

void crossbowExecutionContextSetBatchExamples (crossbowExecutionContextP, int, int *, int);

void crossbowExecutionContextSetBatchLabels (crossbowExecutionContextP, int, int *, int);

void crossbowExecutionContextSetBatchSplits (crossbowExecutionContextP, int);

void crossbowExecutionContextSetRandomSeed (crossbowExecutionContextP, unsigned long long);

int crossbowExecutionContextMaxOperatorsPerDataflow (crossbowExecutionContextP);

void crossbowExecutionContextCreateStreams (crossbowExecutionContextP, int);

void crossbowExecutionContextAddStream (crossbowExecutionContextP);

crossbowStreamP crossbowExecutionContextNextStream (crossbowExecutionContextP, int);

crossbowCallbackHandlerP crossbowExecutionContextNextCallbackHandler (crossbowExecutionContextP);

void crossbowStreamExecute (crossbowExecutionContextP, crossbowStreamP);

void crossbowExecutionContextFree (JNIEnv *, crossbowExecutionContextP);

void crossbowExecutionContextDump (crossbowExecutionContextP);

/* Helpers */

crossbowExecutionContextP crossbowExecutionContextInit (int *, int, int, int, int, int *);

void crossbowExecutionContextSetKernel (crossbowExecutionContextP, int, const char *, int, int, int, int);

void crossbowExecutionContextSetKernelInput (crossbowExecutionContextP, int, int, int, int *, int);

void crossbowExecutionContextSetKernelOutput (crossbowExecutionContextP, int, int, int *, int);

void crossbowExecutionContextSetKernelLocalVariable (crossbowExecutionContextP, int, int, const char *, int, int *, int, int);

void crossbowExecutionContextSetKernelLocalVariableBuffer (crossbowExecutionContextP, int, int, void *);

void crossbowExecutionContextSetKernelScalars (crossbowExecutionContextP, int id, int count);

void crossbowExecutionContextSetKernelScalarAsInt (crossbowExecutionContextP, int, int, const char *, int);

void crossbowExecutionContextSetKernelScalarAsFloat (crossbowExecutionContextP, int, int, const char *, float);

void crossbowExecutionContextSetKernelScalarAsDouble (crossbowExecutionContextP, int, int, const char *, double);

void crossbowExecutionContextCudnnSetKernelType (crossbowExecutionContextP, int, int);

void crossbowExecutionContextCudnnSetKernelInputDescriptor (crossbowExecutionContextP, int, int, int, int, int);

void crossbowExecutionContextCudnnSetKernelOutputDescriptor (crossbowExecutionContextP, int, int, int, int, int);

void crossbowExecutionContextCudnnSetConvolutionDescriptor (crossbowExecutionContextP, int, int, int, int, int);

void crossbowExecutionContextCudnnSetConvolutionFilterDescriptor (crossbowExecutionContextP, int, int, int, int, int);

void crossbowExecutionContextCudnnSetConvolutionBiasDescriptor (crossbowExecutionContextP, int, int, int, int, int);

size_t crossbowExecutionContextCudnnConfigureConvolutionForwardAlgorithm (crossbowExecutionContextP, int, int, double);

size_t crossbowExecutionContextCudnnConfigureConvolutionBackwardFilterAlgorithm (crossbowExecutionContextP, int, int, double);

size_t crossbowExecutionContextCudnnConfigureConvolutionBackwardDataAlgorithm (crossbowExecutionContextP, int, int, double);

void crossbowExecutionContextCudnnSetPoolingMode (crossbowExecutionContextP, int, int);

void crossbowExecutionContextCudnnSetPoolingDescriptor (crossbowExecutionContextP, int, int, int, int, int, int, int);

void crossbowExecutionContextCudnnSetActivationDescriptor (crossbowExecutionContextP, int, int, double);

void crossbowExecutionContextCudnnSetBatchNormDescriptor (crossbowExecutionContextP, int);

void crossbowExecutionContextCudnnSetBatchNormEstimatedMeanAndVariance (crossbowExecutionContextP, int, int);

void crossbowExecutionContextCudnnSetDropoutDescriptor (crossbowExecutionContextP, int, float, unsigned long long);

size_t crossbowExecutionContextCudnnGetDropoutReserveSpaceSize (crossbowExecutionContextP, int);

void crossbowExecutionContextSetKernelConfigurationParameters (crossbowExecutionContextP, int id, int count);

void crossbowExecutionContextSetKernelConfigurationParameterAsInt (crossbowExecutionContextP, int, int, const char *, int);

void crossbowExecutionContextSetKernelConfigurationParameterAsFloat (crossbowExecutionContextP, int, int, const char *, float);

void crossbowExecutionContextSetKernelConfigurationParameterAsIntArray (crossbowExecutionContextP, int, int, const char *, int, int *);

void crossbowExecutionContextSetKernelConfigurationParameterAsFloatArray (crossbowExecutionContextP, int, int, const char *, int, float *);

void crossbowExecutionContextSetKernelConfigurationParameterAsDouble (crossbowExecutionContextP, int, int, const char *, double);

void crossbowExecutionContextSetDataflowGraph (crossbowExecutionContextP, int, int, int *);

void crossbowExecutionContextSetDataflowStream (crossbowExecutionContextP, int, int, int);

void crossbowExecutionContextSetDataflowDependency (crossbowExecutionContextP, int, int, int, int, unsigned);

void crossbowExecutionContextSetDataflowUpstreamNeighbours (crossbowExecutionContextP, int, int, int, int *);

void crossbowExecutionContextSetDataflowDownstreamNeighbours (crossbowExecutionContextP, int, int, int, int *);

void crossbowExecutionContextSetDataflowLossOperator (crossbowExecutionContextP, int, int);

void crossbowExecutionContextSetDataflowAccuracyOperator (crossbowExecutionContextP, int, int);

void crossbowExecutionContextSetDataflowDataTransformOperator (crossbowExecutionContextP, int, int);

void crossbowExecutionContextSetDataflowPeers (crossbowExecutionContextP, int, int, int *);

void crossbowExecutionContextSetDataflowMemoryPlan (crossbowExecutionContextP, int, int, int, int);

void crossbowExecutionContextSetModel (crossbowExecutionContextP, int, int);

void crossbowExecutionContextSetModelVariable (crossbowExecutionContextP, int, int, int, int *, int);

void crossbowExecutionContextSetModelVariableBuffer (crossbowExecutionContextP, int, int, void *);

void crossbowExecutionContextSetModelVariableLearningRateMultiplier (crossbowExecutionContextP, int, int, float);

void crossbowExecutionContextSetModelWorkPerClock (crossbowExecutionContextP, int);

void crossbowExecutionContextSetUpdateModelType (crossbowExecutionContextP, int);

void crossbowExecutionContextSetEamsgdAlpha (crossbowExecutionContextP, float);

void crossbowExecutionContextSetEamsgdTau (crossbowExecutionContextP, int);

void crossbowExecutionContextSetBaseModelMomentum (crossbowExecutionContextP, float);

void crossbowExecutionContextSetMomentum (crossbowExecutionContextP, float, int);

void crossbowExecutionContextSetWeightDecay (crossbowExecutionContextP, float);

void crossbowExecutionContextSetLearningRateDecayPolicyFixed (crossbowExecutionContextP, float);

void crossbowExecutionContextSetLearningRateDecayPolicyInv (crossbowExecutionContextP, float, double, double);

void crossbowExecutionContextSetLearningRateDecayPolicyStep (crossbowExecutionContextP, float, double, int);

void crossbowExecutionContextSetLearningRateDecayPolicyMultiStep (crossbowExecutionContextP, float, double, int, int, int *);

void crossbowExecutionContextSetLearningRateDecayPolicyExp (crossbowExecutionContextP, float, double);

void crossbowExecutionContextSetLearningRateDecayPolicyCircular (crossbowExecutionContextP, float *, int, float *, int);

void crossbowExecutionContextSetModelManager (JNIEnv *, crossbowExecutionContextP, int, int);

jobject crossbowExecutionContextAcquireAccess (JNIEnv *, crossbowExecutionContextP, int *);

jobject crossbowExecutionContextUpgradeAccess (JNIEnv *, crossbowExecutionContextP, jobject, int *);

void crossbowExecutionContextSetResultHandler (crossbowExecutionContextP, int, void *, int);

void crossbowExecutionContextSetLightWeightDatasetHandler (crossbowExecutionContextP, int, void *, int);

void crossbowExecutionContextExecute (
	JNIEnv *,
	crossbowExecutionContextP,
	int,
	int,
	void *, int, int, int,
	void *, int, int, int,
	long *,
	int,
	jobject);

void crossbowExecutionContextExecuteNext (
	JNIEnv *,
	crossbowExecutionContextP,
	int,
	int,
	int, int,
	int, int,
	long *,
	int,
	jobject);

void crossbowExecutionContextSchedule (
	JNIEnv *,
	crossbowExecutionContextP,
	int,
	int,
	void *, int, int, int,
	void *, int, int, int,
	long *,
	int,
	int);

void crossbowExecutionContextScheduleNext (
	JNIEnv *,
	crossbowExecutionContextP,
	int,
	int,
	int, int,
	int, int,
	long *,
	int,
	int);

int crossbowExecutionContextLockModels (crossbowExecutionContextP);

int crossbowExecutionContextMergeModels (crossbowExecutionContextP, int);

int crossbowExecutionContextSynchroniseModels (crossbowExecutionContextP, int, int, int, int);

int crossbowExecutionContextUnlockModels(crossbowExecutionContextP);

int crossbowExecutionContextCheckpointModels (crossbowExecutionContextP, const char *);

int crossbowExecutionContextOverrideModelData (crossbowExecutionContextP, const char *);

int crossbowExecutionContextAddModel (crossbowExecutionContextP);

int crossbowExecutionContextDelModel (crossbowExecutionContextP);

/* Record dataset configuration */

void crossbowExecutionContextRecordDatasetInit (crossbowExecutionContextP, int, int, int *, int, int, int *);

void crossbowExecutionContextRecordDatasetRegister (crossbowExecutionContextP, int, int, const char *);

void crossbowExecutionContextRecordDatasetFinalise (crossbowExecutionContextP, int);

#endif /* __CROSSBOW_EXECUTION_CONTEXT_H_ */
