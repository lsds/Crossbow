#ifndef __CROSSBOW_STREAM_H_
#define __CROSSBOW_STREAM_H_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <cudnn.h>

#include <curand.h>

#include "device.h"
#include "databuffer.h"
#include "variable.h"
#include "list.h"
#include "dataflow.h"
#include "operator.h"
#include "model.h"

#include "utils.h"

typedef struct crossbow_stream *crossbowStreamP;
typedef struct crossbow_stream {
	int id;

	/* Each stream belongs to a particular device */
	int deviceId;
	
	crossbowModelSynchronisationMode_t mode;

	int branches;
	cudaStream_t *stream;

	cublasHandle_t *cublasHandle;
	cudnnHandle_t *cudnnHandle;
	curandGenerator_t curandGenerator;
	
	/* The event that marks the completion of a task */
	cudaEvent_t event;
#ifdef INTRA_TASK_MEASUREMENTS
	cudaEvent_t start;
#endif

#ifdef MAKESPAN_MEASUREMENTS
	cudaEvent_t barrier;
#endif

    int splits;
    
	crossbowDataBufferP input;

	crossbowVariableP examples;
	crossbowVariableP labels;

	int ops;
	/*
	 * A kernel can produce more than one output buffers,
	 * which will be appended to a corresponding list.
	 */
	crossbowListP *outputs;

	/*
	 * A list of local variables used by operators in the
	 * dataflow that are not read-only and need to return
	 * to their corresponding queue.
	 *
	 * [Update 12/3] There a list per operator.
	 */
	crossbowListP *locals;

	int task;
	crossbowPhase_t phi; /* If CHECK, dataflow execution returns accuracy results */

	long freeP[2];

	crossbowDataflowP dataflow;
	crossbowOperatorP op;

	crossbowModelP model;

	/* Attributes associated with the parameter server synchronisation model */
	crossbowModelP theModel;
	cublasHandle_t modelSynchronisationHandle;
	cudaStream_t modelSynchronisationStream;

} crossbow_stream_t;

crossbowStreamP crossbowStreamCreate (int, crossbowDeviceP, int, int, crossbowVariableSchemaP, crossbowVariableSchemaP, int, crossbowModelSynchronisationMode_t, unsigned long long);

crossbowDataBufferP crossbowStreamOperatorGetInput (crossbowStreamP, crossbowOperatorP);

crossbowDataBufferP crossbowStreamOperatorGetOutput (crossbowStreamP, crossbowOperatorP);

crossbowDataBufferP crossbowStreamGetCurrentInput (crossbowStreamP);

crossbowDataBufferP crossbowStreamGetPeerInput (crossbowStreamP);

crossbowDataBufferP crossbowStreamGetCurrentOutput (crossbowStreamP);

crossbowDataBufferP crossbowStreamGetPeerOutput (crossbowStreamP);

void crossbowStreamComputeInputCheckSum (crossbowStreamP);

void crossbowStreamComputeCheckSum (crossbowStreamP);

void crossbowStreamUpdateModel (crossbowStreamP);

void crossbowStreamClear (crossbowStreamP);

void crossbowStreamFree (crossbowStreamP);

#endif /* __CROSSBOW_STREAM_H_ */
