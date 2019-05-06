#include "executioncontext.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include "stream.h"
#include "kernel.h"

#include "dataflow.h"

#include "operatordependency.h"

#include "modelmanager.h"

#include "callbackhandler.h"
#include "taskhandler.h"

#include "resulthandler.h"

#include "kernelmap.h"

#include "localvariable.h"
#include "kernelconfigurationparameter.h"
#include "kernelscalar.h"

#include "kernels/noop.h"
#include "kernels/noopstateless.h"
#include "kernels/matmul.h"
#include "kernels/innerproduct.h"
#include "kernels/softmax.h"
#include "kernels/softmaxloss.h"
#include "kernels/conv.h"
#include "kernels/relu.h"
#include "kernels/pool.h"
#include "kernels/dropout.h"
#include "kernels/lrn.h"
#include "kernels/batchnorm.h"
#include "kernels/elementwiseop.h"
#include "kernels/concat.h"

#include "kernels/cudnnconv.h"
#include "kernels/cudnnrelu.h"
#include "kernels/cudnnpool.h"
#include "kernels/cudnnsoftmax.h"
#include "kernels/cudnnbatchnorm.h"
#include "kernels/cudnndropout.h"

#include "kernels/cudnnconvgradient.h"
#include "kernels/cudnnrelugradient.h"
#include "kernels/cudnnpoolgradient.h"
#include "kernels/cudnnsoftmaxgradient.h"
#include "kernels/cudnnbatchnormgradient.h"
#include "kernels/cudnndropoutgradient.h"

#include "kernels/concatgradient.h"
#include "kernels/elementwiseopgradient.h"
#include "kernels/batchnormgradient.h"
#include "kernels/lrngradient.h"
#include "kernels/dropoutgradient.h"
#include "kernels/poolgradient.h"
#include "kernels/relugradient.h"
#include "kernels/convgradient.h"
#include "kernels/softmaxlossgradient.h"
#include "kernels/softmaxgradient.h"
#include "kernels/innerproductgradient.h"

#include "kernels/gradientdescentoptimiser.h"
#include "kernels/accuracy.h"
#include "kernels/classify.h"

#include "kernels/datatransform.h"

#include "kernels/matfact.h"

#include "kernels/sleep.h"

#include "cudnn/cudnnbatchnormparams.h"

#include "image/yarng.h"

#include "synch/sma.h"
#include "synch/eamsgd.h"
#include "synch/default.h"
#include "synch/hogwild.h"
#include "synch/downpour.h"
#include "synch/polyakruppert.h"
#include "synch/synchronoussgd.h"
#include "synch/synchronouseamsgd.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <math.h>

/*
 * Based on <helper_cuda.h>
 */
static inline int __convert_sm_to_cores (int major, int minor) {

	typedef struct {
		int sm; /* 0xMm (hexidecimal notation) where `M` is SM major version and `m` is SM minor version */
		int cores;
	} __sm_mapping_t;

	int index;

	__sm_mapping_t map [] = {
		{ 0x20,  32}, // Fermi   Generation (SM 2.0) GF100 class
		{ 0x21,  48}, // Fermi   Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler  Generation (SM 3.0) GK10x class
		{ 0x32, 192}, // Kepler  Generation (SM 3.2) GK10x class
		{ 0x35, 192}, // Kepler  Generation (SM 3.5) GK11x class
		{ 0x37, 192}, // Kepler  Generation (SM 3.7) GK21x class
		{ 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x60,  64}, // Pascal  Generation (SM 6.0) GP100 class
		{ 0x61, 128}, // Pascal  Generation (SM 6.1) GP10x class
		{ 0x62, 128}, // Pascal  Generation (SM 6.2) GP10x class
		{ 0x70,  64}, // Volta   Generation (SM 7.0) GV100 class
		{   -1,  -1}  // Exit
	};

	index = 0;
	while (map [index].sm != -1) {
		if (map [index].sm == ((major << 4) + minor))
			return map [index].cores;
		index++;
	}

	warn("SM %d.%d is undefined - defaults to %d cores\n", major, minor, map [index - 1].cores);
	return map [index-1].cores;
}

/*
static void CUDART_CB callback (cudaStream_t stream, cudaError_t error, void *args) {
#ifdef GPU_VERBOSE
	crossbowStreamP s = (crossbowStreamP) args;
	dbg ("CUDA stream %p, crossbow stream %p\n", stream, s->stream);
	dbg("Task %d finished\n", s->task);
#else
	(void) stream;
#endif
	checkCudaErrors(error);
	crossbowCallbackHandlerP handler = crossbowExecutionContextNextCallbackHandler (theGPU);
	crossbowCallbackHandlerPublish (handler, args);
	return;
}
*/

crossbowExecutionContextP crossbowExecutionContextCreate (crossbowArrayListP devices, int numberofstreams,
		int numberofcallbackhandlers, int numberoftaskhandlers, int *offset) {

	crossbowDeviceP dev;
	int ndx; /* Generic iterator */

	crossbowExecutionContextP p;
	p = (crossbowExecutionContextP) crossbowMalloc (sizeof(crossbow_execution_context_t));

	memset(p, 0, sizeof(crossbow_execution_context_t));

	crossbowMemoryManagerInit ();

	p->batch = crossbowBatchCreate ();

	/* Array of devices has already been initialised */
	p->devices = devices;

	/* Set first available device as the default device */
	p->defaultDeviceId = -1;

	int __count = 0;
	for (ndx = 0; ndx < crossbowArrayListSize (p->devices); ++ndx) {
		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, ndx);
		if (crossbowDeviceSelected (dev)) {
			__count ++;
			if (p->defaultDeviceId < 0)
				p->defaultDeviceId = dev->id;
		}
	}
	invalidConditionException(p->defaultDeviceId >= 0);
	p->mode = (__count > 1) ? MULTI_GPU : SINGLE_GPU;

#ifdef USE_NCCL
	/* Configure NCCL library */
	p->nc = __count;

	p->comms = (ncclComm_t *) crossbowMalloc(p->nc * sizeof(ncclComm_t));

	p->devs = (int *) crossbowMalloc(p->nc * sizeof(int));
	int j = 0;
	for (ndx = 0; ndx < crossbowArrayListSize (p->devices); ++ndx) {
		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, ndx);
		if (crossbowDeviceSelected (dev))
			p->devs[j++] = dev->id;
	}
	invalidConditionException(j == __count);

	checkNcclErrors(ncclCommInitAll(p->comms, __count, p->devs));
#endif

	/*
	 * Create `streams` an an array of queues, one per device.
	 * There are multiple streams per device.
	 */
	p->streams = crossbowArrayListCreate (crossbowArrayListSize (p->devices));
	for (ndx = 0; ndx < crossbowArrayListSize (p->devices); ++ndx) {
		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, ndx);
		if (! crossbowDeviceSelected (dev))
			continue;
		crossbowArrayListSet (p->streams, dev->id, crossbowThetaQueueCreate (numberofstreams));
	}

	p->previousStreamId = -1;

	p->dataflows = crossbowArrayListCreate (0);
	p->kernels   = crossbowArrayListCreate (0);

	p->theModel = NULL;
	p->modelmanager = NULL;

	p->resulthandler = crossbowResultHandlerCreate (64); /* Result slots are 64-byte wide */

	/* This will be set only if we are using light-weight datasets */
	p->datasethandler = NULL;

	p->kernelmap = NULL;

	/* Initialise callback handlers */
    /*
	p->callbackhandlers = crossbowArrayListCreate (numberofcallbackhandlers);
	for (ndx = 0; ndx < numberofcallbackhandlers; ++ndx)
		crossbowArrayListSet (p->callbackhandlers, ndx, 
			crossbowCallbackHandlerCreate(p->modelmanager, p->resulthandler, p->streams, offset[0]));
    */

	(void) offset;

    int cb_sockets = 2;
    invalidConditionException((numberofcallbackhandlers % cb_sockets) == 0);

    int callbackhandlerspersocket = numberofcallbackhandlers / cb_sockets;
    p->callbackhandlers = crossbowArrayListCreate (cb_sockets);
    int kcb;
    int corecb;
    for (ndx = 0; ndx < cb_sockets; ++ndx) {
        crossbowArrayListP array = crossbowArrayListCreate(callbackhandlerspersocket);
        for (kcb = 0; kcb < callbackhandlerspersocket; ++kcb) {
            corecb = (ndx == 0) ? (6 + kcb) : (14 + kcb);
            crossbowArrayListSet(array, kcb, 
                crossbowCallbackHandlerCreate(p->modelmanager, p->resulthandler, p->streams, corecb));
        }
        crossbowArrayListSet (p->callbackhandlers, ndx, array);
    }
	/* Await initialisation of callback handlers */
	
#ifdef USE_TASKHANDLERS
	/*
	 * Update (31 May 2018)
	 *
	 * Task handlers, once pinned to a CPU core, were able to schedule tasks to any GPU.
	 * But this did not guarantee optimal GPU affinity.
	 *
	 * invalidConditionException (numberoftaskhandlers > 0);
	 * p->taskhandlers = crossbowArrayListCreate (numberoftaskhandlers);
	 * for (ndx = 0; ndx < numberoftaskhandlers; ++ndx)
	 * 	crossbowArrayListSet (p->taskhandlers, ndx, crossbowTaskHandlerCreate (p->callbackhandlers, offset[1], 0));
     */

	/*
	 * The new version of the task handler
	 * assumes that there are two sockets.
	 */
	
	int sockets = 2;
	invalidConditionException((numberoftaskhandlers % sockets) == 0);

	int taskhandlerspersocket = numberoftaskhandlers / sockets;

	p->taskhandlers = crossbowArrayListCreate (sockets);
	int k;
	int core;
	for (ndx = 0; ndx < sockets; ++ndx) {
		crossbowArrayListP array = crossbowArrayListCreate(taskhandlerspersocket);
		for (k = 0; k < taskhandlerspersocket; ++k) {
			core = (ndx == 0) ? (2 + k) : (10 + k);
			crossbowArrayListSet(array, k, crossbowTaskHandlerCreate (p->callbackhandlers, ndx, core));
		}
		crossbowArrayListSet (p->taskhandlers, ndx, array);
	}

#else
	(void) numberoftaskhandlers;
#endif

	/* cuBLAS and cuDNN handler(s)
	 *
	 * These handlers, one per device, will be used by the GPU worker thread, responsible
	 * for task execution.
	 *
	 * A different handle will be created for the result collector, responsible for model
	 * synchronisation on a particular device.
	 */
	for (ndx = 0; ndx < crossbowArrayListSize (p->devices); ++ndx) {
		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, ndx);
		if (! crossbowDeviceSelected (dev))
			continue;
		/* Redirect CUDA calls to specific device */
		checkCudaErrors(cudaSetDevice(dev->id));
		#ifdef MAKESPAN_MEASUREMENTS
		// checkCudaErrors(cudaEventCreateWithFlags(&(dev->barrier), cudaEventBlockingSync));
		checkCudaErrors(cudaEventCreateWithFlags(&(dev->barrier), cudaEventDefault));
		#endif
		/* cuBLAS handler */
		checkCublasStatus(cublasCreate(&(dev->cublasHandle)));
		/* cuDNN handler */
		checkCudnnStatus(cudnnCreate(&(dev->cudnnHandle)));
		/* cuRAND generator */
		checkCurandStatus(curandCreateGenerator(&(dev->curandGenerator), CURAND_RNG_PSEUDO_DEFAULT));
		/* Create cuBLAS handle for synchronisation */
		checkCublasStatus(cublasCreate(&(dev->modelSynchronisationHandle)));
		/* Create model synchronisation stream */
		checkCudaErrors(cudaStreamCreateWithFlags(&(dev->modelSynchronisationStream), cudaStreamNonBlocking));
		/* Assign synchronisation stream to handle */
		checkCublasStatus(cublasSetStream(dev->modelSynchronisationHandle, dev->modelSynchronisationStream));
	}

	/* Reset to default device */
	checkCudaErrors(cudaSetDevice(p->defaultDeviceId));
	
#ifdef INTER_TASK_MEASUREMENTS
	p->timer = crossbowTimerCreate ();
	p->measurements = crossbowMeasurementListCreate (64, 1);
#endif

	p->seed = 0ULL;

	p->version = 0ULL;

	p->dataset[0] = NULL;
	p->dataset[1] = NULL;

	return p;
}

void crossbowExecutionContextBindKernels (crossbowExecutionContextP p) {

	p->kernelmap = crossbowKernelMapCreate (100); /* 100 slots */

	crossbowKernelMapBind (p->kernelmap, "Noop",                                         crossbowKernelNoop);
	crossbowKernelMapBind (p->kernelmap, "NoopStateless",                       crossbowKernelNoopStateless);
	crossbowKernelMapBind (p->kernelmap, "MatMul",                                     crossbowKernelMatMul);

	crossbowKernelMapBind (p->kernelmap, "DataTransform",                       crossbowKernelDataTransform);

	crossbowKernelMapBind (p->kernelmap, "InnerProduct",                         crossbowKernelInnerProduct);
#ifndef USE_CUDNN
	crossbowKernelMapBind (p->kernelmap, "SoftMax",                                   crossbowKernelSoftMax);
	crossbowKernelMapBind (p->kernelmap, "Conv",                                         crossbowKernelConv);
	crossbowKernelMapBind (p->kernelmap, "ReLU",                                         crossbowKernelReLU);
	crossbowKernelMapBind (p->kernelmap, "Pool",                                         crossbowKernelPool);
	crossbowKernelMapBind (p->kernelmap, "BatchNorm",                               crossbowKernelBatchNorm);
	crossbowKernelMapBind (p->kernelmap, "Dropout",                                   crossbowKernelDropout);
#else
	crossbowKernelMapBind (p->kernelmap, "SoftMax",                              crossbowCudnnKernelSoftMax);
	crossbowKernelMapBind (p->kernelmap, "Conv",                                    crossbowCudnnKernelConv);
	crossbowKernelMapBind (p->kernelmap, "ReLU",                                    crossbowCudnnKernelReLU);
	crossbowKernelMapBind (p->kernelmap, "Pool",                                    crossbowCudnnKernelPool);
	crossbowKernelMapBind (p->kernelmap, "BatchNorm",                          crossbowCudnnKernelBatchNorm);
	crossbowKernelMapBind (p->kernelmap, "Dropout",                              crossbowCudnnKernelDropout);
#endif
	crossbowKernelMapBind (p->kernelmap, "SoftMaxLoss",                           crossbowKernelSoftMaxLoss);
	crossbowKernelMapBind (p->kernelmap, "LRN",                                           crossbowKernelLRN);
	crossbowKernelMapBind (p->kernelmap, "ElementWiseOp",                       crossbowKernelElementWiseOp);
	crossbowKernelMapBind (p->kernelmap, "Concat",                                     crossbowKernelConcat);

	crossbowKernelMapBind (p->kernelmap, "InnerProductGradient",         crossbowKernelInnerProductGradient);
	crossbowKernelMapBind (p->kernelmap, "SoftMaxLossGradient",           crossbowKernelSoftMaxLossGradient);
#ifndef USE_CUDNN
	crossbowKernelMapBind (p->kernelmap, "SoftMaxGradient",                   crossbowKernelSoftMaxGradient);
	crossbowKernelMapBind (p->kernelmap, "ConvGradient",                         crossbowKernelConvGradient);
	crossbowKernelMapBind (p->kernelmap, "ReLUGradient",                         crossbowKernelReLUGradient);
	crossbowKernelMapBind (p->kernelmap, "PoolGradient",                         crossbowKernelPoolGradient);
	crossbowKernelMapBind (p->kernelmap, "BatchNormGradient",               crossbowKernelBatchNormGradient);
	crossbowKernelMapBind (p->kernelmap, "DropoutGradient",                   crossbowKernelDropoutGradient);
#else
	crossbowKernelMapBind (p->kernelmap, "SoftMaxGradient",              crossbowCudnnKernelSoftMaxGradient);
	crossbowKernelMapBind (p->kernelmap, "ConvGradient",                    crossbowCudnnKernelConvGradient);
	crossbowKernelMapBind (p->kernelmap, "ReLUGradient",                    crossbowCudnnKernelReLUGradient);
	crossbowKernelMapBind (p->kernelmap, "PoolGradient",                    crossbowCudnnKernelPoolGradient);
	crossbowKernelMapBind (p->kernelmap, "BatchNormGradient",          crossbowCudnnKernelBatchNormGradient);
	crossbowKernelMapBind (p->kernelmap, "DropoutGradient",              crossbowCudnnKernelDropoutGradient);
#endif
	crossbowKernelMapBind (p->kernelmap, "LRNGradient",                           crossbowKernelLRNGradient);
	crossbowKernelMapBind (p->kernelmap, "ElementWiseOpGradient",       crossbowKernelElementWiseOpGradient);
	crossbowKernelMapBind (p->kernelmap, "ConcatGradient",                     crossbowKernelConcatGradient);

	crossbowKernelMapBind (p->kernelmap, "GradientDescentOptimiser", crossbowKernelGradientDescentOptimiser);
	crossbowKernelMapBind (p->kernelmap, "Accuracy",                                 crossbowKernelAccuracy);
	crossbowKernelMapBind (p->kernelmap, "Classify",                                 crossbowKernelClassify);

	crossbowKernelMapBind (p->kernelmap, "MatFact",                                   crossbowKernelMatFact);
#ifdef GPU_VERBOSE
	crossbowKernelMapDump (p->kernelmap);
#endif
}

void crossbowExecutionContextSetBatchExamples (crossbowExecutionContextP p, int dims, int *shape, int bytes) {
	crossbowVariableSchemaP examples = crossbowVariableSchemaCreate  (dims, shape, bytes);
	crossbowBatchSetExampleSchema (p->batch, examples);
	return;
}

void crossbowExecutionContextSetBatchLabels (crossbowExecutionContextP p, int dims, int *shape, int bytes) {
	crossbowVariableSchemaP labels = crossbowVariableSchemaCreate  (dims, shape, bytes);
	crossbowBatchSetLabelSchema (p->batch, labels);
	return;
}

void crossbowExecutionContextSetBatchSplits (crossbowExecutionContextP p, int splits) {
	crossbowBatchSetSplits (p->batch, splits);
    return;
}

int crossbowExecutionContextMaxOperatorsPerDataflow (crossbowExecutionContextP p) {
	int i;
	crossbowDataflowP dataflow;
	int ops, max = 0;
	for (i = 0; i < crossbowArrayListSize (p->dataflows); i++) {
		dataflow = (crossbowDataflowP) crossbowArrayListGet (p->dataflows, i);
		ops = crossbowDataflowSize(dataflow);
		if (max < ops)
			max = ops;
	}
	return max;
}

void crossbowExecutionContextCreateStreams (crossbowExecutionContextP p, int branches) {

	int deviceId; /* Device iterator */
	crossbowDeviceP dev;

	int ndx;
	int size;
	crossbowThetaQueueP queue;

	int ops;
	int splits;

	crossbowVariableSchemaP examples, labels;
	crossbowStreamP stream;

	ops = crossbowExecutionContextMaxOperatorsPerDataflow (p);

	examples = crossbowBatchGetExampleSchema (p->batch);
	labels   = crossbowBatchGetLabelSchema   (p->batch);
	splits   = crossbowBatchGetSplits        (p->batch);

	for (deviceId = 0; deviceId < crossbowArrayListSize (p->devices); ++deviceId) {
		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, deviceId);
		if (! crossbowDeviceSelected(dev))
			continue;
		/* Get device-specific queue */
		queue = (crossbowThetaQueueP) crossbowArrayListGet (p->streams, dev->id);
		size = crossbowThetaQueueSize (queue);
		/* Note that schemas are replicated in `crossbowStreamCreate` */
		for (ndx = 0; ndx < size; ndx++) {
			stream = crossbowStreamCreate (ndx, dev, ops, splits, examples, labels, branches, p->mode, p->seed);
			crossbowThetaQueueSet (queue, ndx, (void *) stream);
		}
	}
	return;
}

void crossbowExecutionContextAddStream (crossbowExecutionContextP p) {
	int deviceId; /* Device iterator */
	crossbowDeviceP dev;

	int id;
	crossbowThetaQueueP queue;

	int ops;
	int splits;

	crossbowVariableSchemaP examples, labels;
	crossbowStreamP stream;

	/* char *str; */

	int i, j;
	crossbowKernelP kernel;
	crossbowLocalVariableP variable;

	ops = crossbowExecutionContextMaxOperatorsPerDataflow (p);

	examples = crossbowBatchGetExampleSchema (p->batch);
	labels   = crossbowBatchGetLabelSchema   (p->batch);
	splits   = crossbowBatchGetSplits        (p->batch);

	for (deviceId = 0; deviceId < crossbowArrayListSize (p->devices); ++deviceId) {
		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, deviceId);
		if (! crossbowDeviceSelected(dev))
			continue;

		/* Get device-specific queue */
		queue = (crossbowThetaQueueP) crossbowArrayListGet (p->streams, dev->id);
		id = crossbowThetaQueueSize (queue);
		/*
		 * TODO: Currently code assumes 1 branch per execution stream. Get
		 * the number of branches from an already instantiated stream.
		 */
		stream = crossbowStreamCreate (id, dev, ops, splits, examples, labels, 1, p->mode, p->seed);
		crossbowThetaQueueExpand (queue, (void *) stream);
	}

	/* Go over kernel and resize their outputs and local variables appropriately */
	for (i = 0; i < crossbowArrayListSize (p->kernels); ++i) {

		kernel = (crossbowKernelP) crossbowArrayListGet (p->kernels, i);
		nullPointerException (kernel);

		crossbowKernelResizeOutputBufferPool (kernel, p->devices);

		// Does the kernel have any local variables?
		if (kernel->variables) {

			for (j = 0; j < crossbowArrayListSize (kernel->variables); ++j) {

				variable = crossbowArrayListGet (kernel->variables, j);
                /* It is unlikely that a variable is NULL, unless
                 * a kernel is misconfigured. */
                if (! variable)
                    continue;
				// nullPointerException (variable);
				if (! crossbowLocalVariableReadOnly (variable))
					crossbowLocalVariableResizePool (variable, p->devices);
			}
		}
	}
}


crossbowStreamP crossbowExecutionContextNextStream (crossbowExecutionContextP p, int dev) {
	crossbowThetaQueueP queue = (crossbowThetaQueueP) crossbowArrayListGet (p->streams, dev);
	/* Blocks until a stream becomes available */
	crossbowStreamP stream = (crossbowStreamP) crossbowThetaQueueGetNext (queue);
	nullPointerException (stream);
	return stream;
}

crossbowCallbackHandlerP crossbowExecutionContextNextCallbackHandler (crossbowExecutionContextP p) {
	crossbowCallbackHandlerP handler = (crossbowCallbackHandlerP) crossbowArrayListGetNext (p->callbackhandlers);
	nullPointerException (handler);
	return handler;
}

void crossbowStreamExecute (crossbowExecutionContextP ctx, crossbowStreamP s) {
	
	crossbowOperatorDependencyP d;
	
#ifdef __NVPROF_MARK_TASK
	nvtxMarkA("crossbowTaskSubmit");
#endif

#ifdef INTER_TASK_MEASUREMENTS
	/* Next task is ready to execute: get time delta between two consecutive tasks */
	if (! crossbowTimerRunning (ctx->timer)) {
		crossbowTimerStart (ctx->timer);
	}
	else {
		tstamp_t dt = crossbowTimerLap (ctx->timer);
		crossbowMeasurementListAppend (ctx->measurements, ((float) dt));
		if (s->task % INTER_TASK_MEASUREMENTS_DISPLAY_INTERVAL == 0) {
			info("Task inter-arrival time is %13.3f usecs\n", crossbowMeasurementListRunningAverage(ctx->measurements));
		}
	}
#endif

	/* All CUDA calls already redirected to specific device */

#ifdef MAKESPAN_MEASUREMENTS
	if (s->task == 1) {
		int64_t seconds = 10;
		info("First task, delay execution of all subsequent tasks by %ld seconds\n", seconds);
		crossbowDeviceP dev = (crossbowDeviceP) crossbowArrayListGet (ctx->devices, s->deviceId);
		int64_t cycles = dev->frequency * seconds;
		crossbowKernelSleep (&cycles);
		checkCudaErrors(cudaEventRecord(s->barrier, NULL));
	}
	info("Schedule task %d on device %d stream %d\n", s->task, s->deviceId, s->id);
#endif

	dbg("Current stream is %d, previous stream is %d\n", s->id, ctx->previousStreamId);

	/* Double-check that there are no pending asynchronous tasks on this CUDA streams */
	checkCudaErrors(cudaEventQuery(s->event));

#ifdef MAKESPAN_MEASUREMENTS
	for (i = 0; i < s->branches; i++)
		checkCudaErrors(cudaStreamWaitEvent(s->stream[i], s->barrier, 0));
#endif

#ifdef INTRA_TASK_MEASUREMENTS
	/* Record start of task */
	checkCudaErrors(cudaEventRecord(s->start, s->stream[0]));
#endif

	/* Input data movement to stream zero */
#ifdef __INPUT_ISPINNED_
	/* Push input variables in one go, since the input data have been copied in continuous memory regions */
	crossbowDataBufferPush (s->input, s->stream[0]);
#else
	/* Input variables have to be copied one-by-one */
	crossbowVariablePush (s->examples, s->stream[0]);
	crossbowVariablePush (s->labels,   s->stream[0]);
#endif
	
	/* Set cuRAND stream.
	 *
	 * cuRAND is used by the data transformation operator, so we set the stream accordingly.
	 * Nonetheless, we don't record dependencies for data movement.
	 */
	if (s->dataflow->dataTransformOp)
		checkCurandStatus(curandSetStream(s->curandGenerator, s->stream[s->dataflow->dataTransformOp->branch]));

	/* The cuBLAS and cuDNN streams are set appropriately at creation time. */

	/* Iterate over dataflow operators and schedule kernels */
	s->op = crossbowDataflowPeek (s->dataflow);
	while (s->op != s->dataflow->tail) {
		dbg("Schedule kernel function(s) for op %s\n", s->op->kernel->name);

		#ifdef __NVPROF_MARK_OPERATOR
		nvtxMarkA("crossbowOperatorBegins");
		#endif

		/* === Optional logic ===
		 *
		 * Record the begining of current operator:
		 *
		 * checkCudaErrors(cudaEventRecord(s->op->start[s->id], s->stream[s->op->branch]));
		 */

		/* === Optional logic ===
		 *
		 * Imposes a strict execution order for the same operators across different tasks.
		 *
		 * The current task operator can not start until the corresponding operator in the
		 * previous task has not finished:
		 *
		 *
	 	 * if (ctx->previousStreamId >= 0)
		 * 	checkCudaErrors(cudaStreamWaitEvent(s->stream, s->op->end[ctx->previousStreamId], 0));
		 */

		/* Intra-task dependencies */
		if (s->branches > 1 && (! crossbowListEmpty (s->op->deps))) {
			crossbowListIteratorReset (s->op->deps);
			while (crossbowListIteratorHasNext(s->op->deps)) {
				d = (crossbowOperatorDependencyP) crossbowListIteratorNext (s->op->deps);
				checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], d->guard->end[s->id], 0));
			}
		}

		crossbowOperatorSchedule (s->op, s); /* Schedule current operator's kernel */

		if (s->branches > 1) {
			if (s->op->required > 0 || crossbowOperatorIsMostDownstream (s->op))
				checkCudaErrors(cudaEventRecord(s->op->end[s->id], s->stream[s->op->branch]));
		}

		#ifdef __NVPROF_MARK_OPERATOR
		nvtxMarkA("crossbowOperatorEnds");
		#endif

		#ifdef UPDATE_MODEL_INCREMENTALLY
		crossbowStreamUpdateModel (s);
		#endif

		#ifdef COMPUTE_CHECKSUM
		crossbowStreamComputeCheckSum (s); /* Compute current operator's checksum  */
		#endif

		s->op = s->op->next; /* Get next operator */
	}

	/*
	 * Making sure that all operators have finished before scheduling any output data movement.
	 * The assumption here is that we can wait for all downstream operators to finish.
	 */
	if (s->branches > 1) {
		s->op = crossbowDataflowPeek (s->dataflow);
		while (s->op != s->dataflow->tail) {
			if (crossbowOperatorIsMostDownstream (s->op))
				checkCudaErrors(cudaStreamWaitEvent(s->stream[0], s->op->end[s->id], 0));
			s->op = s->op->next; /* Get next operator */
		}
	}

	/* Iterate once again, this time scheduling any output data movement */
	s->op = crossbowDataflowPeek (s->dataflow);
	while (s->op != s->dataflow->tail) {
		if (crossbowKernelOutputPull(s->op->kernel)) {
			dbg("Schedule output data movement operation(s) for op %s\n", s->op->kernel->name);
			// crossbowDataBufferPull(crossbowListPeekHead(s->outputs[s->op->id]), s->stream[s->branches - 1]);
			crossbowDataBufferPull(crossbowListPeekHead(s->outputs[s->op->id]), s->stream[0]);
		}
		s->op = s->op->next;
	}

	/* Record task completion event */
	checkCudaErrors(cudaEventRecord(s->event, s->stream[0]));
	
#ifdef __NVPROF_MARK_TASK
	nvtxMarkA("crossbowTaskSubmitted");
#endif
	
	ctx->previousStreamId = s->id;

	/* Handle task completion event */
	crossbowCallbackHandlerP handler = crossbowExecutionContextNextCallbackHandler (ctx);
	crossbowCallbackHandlerPublish (handler, s);

	return;
}

void crossbowExecutionContextFree (JNIEnv *env, crossbowExecutionContextP p) {
	int i, j;
	int size;
    
    crossbowArrayListP pool;
	crossbowThetaQueueP queue;
	crossbowStreamP stream;
	crossbowDataflowP dataflow;
	crossbowKernelP kernel;
	crossbowDeviceP dev;
	crossbowCallbackHandlerP callbackhandler;
	crossbowTaskHandlerP taskhandler;

#ifdef INTER_TASK_MEASUREMENTS
	crossbowTimerFree (p->timer);
	info("%d task inter-arrival measurements, %7.2f us per task\n",
		crossbowMeasurementListElements(p->measurements), crossbowMeasurementListAverage(p->measurements));
	crossbowMeasurementListFree (p->measurements);
#endif

	/* Free callback handlers */
	dbg("Free callback handlers\n");
	size = crossbowArrayListSize (p->callbackhandlers);
#ifdef INTRA_TASK_MEASUREMENTS
	printf("=== [Task measurements from %d callback handlers] ===\n", size);
#endif
	for (i = 0; i < crossbowArrayListSize (p->callbackhandlers); i++) {
        /* Get pool of handlers for a particular socket */
        pool = crossbowArrayListGet (p->callbackhandlers, i);
        for (j = 0; j < crossbowArrayListSize(pool); j++) {
		    callbackhandler = (crossbowCallbackHandlerP) crossbowArrayListGet (pool, j);
#ifdef INTRA_TASK_MEASUREMENTS
		    printf("Callback handler #%02lu: %3d tasks %7.2f ms/task\n",
			    callbackhandler->id, 
                crossbowMeasurementListElements(callbackhandler->measurements), 
                crossbowMeasurementListAverage(callbackhandler->measurements));
#endif
		    crossbowCallbackHandlerFree (callbackhandler);
        }
        crossbowArrayListFree(pool);
	}
#ifdef INTRA_TASK_MEASUREMENTS
	printf("=== [End of task measurements] ===\n");
	fflush(stdout);
#endif
	crossbowArrayListFree (p->callbackhandlers);

#ifdef USE_TASKHANDLERS
	dbg("Free task handlers\n");
	for (i = 0; i < crossbowArrayListSize (p->taskhandlers); i++) {
        /* Get pool of handlers for a particular socket */
        pool = crossbowArrayListGet (p->taskhandlers, i);
        for (j = 0; j < crossbowArrayListSize(pool); j++) {
		    taskhandler = (crossbowTaskHandlerP) crossbowArrayListGet (pool, j);
		    crossbowTaskHandlerFree (taskhandler);
        }
        crossbowArrayListFree(pool);
	}
	crossbowArrayListFree (p->taskhandlers);
#else
	(void) taskhandler;
#endif
    
	dbg("Free stream(s)\n");
	size = crossbowArrayListSize (p->streams);
	for (i = 0; i < size; i++) {
		queue = (crossbowThetaQueueP) crossbowArrayListGet (p->streams, i);
		if (queue) {
			for (j = 0; j < crossbowThetaQueueSize(queue); ++j) {
				stream = crossbowThetaQueueGet (queue, j);
				crossbowStreamFree (stream);
			}
			crossbowThetaQueueFree (queue);
		}
	}
	crossbowArrayListFree (p->streams);

	dbg("Free dataflow(s)\n");
	size = crossbowArrayListSize (p->dataflows);
	for (i = 0; i < size; i++) {
		dataflow = (crossbowDataflowP) crossbowArrayListGet (p->dataflows, i);
		crossbowDataflowFree (dataflow);
	}
	crossbowArrayListFree (p->dataflows);

	dbg("Free kernel(s)\n");
	size = crossbowArrayListSize (p->kernels);
	for (i = 0; i < size; i++) {
		kernel = (crossbowKernelP) crossbowArrayListGet (p->kernels, i);
		dbg("Free kernel %s\n", kernel->name);
		crossbowKernelFree (kernel);
		dbg("Done\n");
	}
	crossbowArrayListFree (p->kernels);

	/* Free model(s) */
	dbg("Free model(s)\n");
	crossbowModelManagerFree (env, p->modelmanager);

	/* Free result slots */
	dbg("Free result handler\n");
	crossbowResultHandlerFree (p->resulthandler);

	/* Free light-weight dataset slots */
	if (p->datasethandler) {
		dbg("Free light-weight dataset handler\n");
		crossbowLightWeightDatasetHandlerFree (p->datasethandler);
	}

	/* Free kernel map */
	dbg("Free kernel map\n");
	crossbowKernelMapFree (p->kernelmap);

	/* Free batch */
	dbg("Free batch\n");
	crossbowBatchFree (p->batch);

	/* Free device(s) */
	dbg("Free devices(s)\n");
	size = crossbowArrayListSize (p->devices);
	for (i = 0; i < size; i++) {

		dev = (crossbowDeviceP) crossbowArrayListGet (p->devices, i);
		dbg("Free device %d\n", dev->id);

		if (crossbowDeviceSelected (dev)) {

#ifdef MAKESPAN_MEASUREMENTS
			/* Free barrier */
			checkCudaErrors(cudaEventDestroy(dev->barrier));
#endif
			/* Free device handlers */
			checkCublasStatus(cublasDestroy(dev->cublasHandle));
			checkCudnnStatus(cudnnDestroy(dev->cudnnHandle));
			checkCurandStatus(curandDestroyGenerator(dev->curandGenerator));

			/* Free synchronisation handle/stream */
			checkCublasStatus(cublasDestroy(dev->modelSynchronisationHandle));
			checkCudaErrors(cudaStreamDestroy(dev->modelSynchronisationStream));
		}

		crossbowDeviceFree (dev);
	}
	crossbowArrayListFree (p->devices);

#ifdef USE_NCCL
	/* Free NCCL context */
	for(int i = 0; i < p->nc; ++i)
		ncclCommDestroy(p->comms[i]);

	crossbowFree(p->comms, p->nc * sizeof(ncclComm_t));
	crossbowFree(p->devs,  p->nc * sizeof(int));
#endif

	if (p->dataset[0])
		crossbowRecordDatasetFree (p->dataset[0]);

	if (p->dataset[1])
		crossbowRecordDatasetFree (p->dataset[1]);

	/* Free execution context */
	crossbowFree(p, sizeof(crossbow_execution_context_t));

	crossbowMemoryManagerDestroy ();

	crossbowMemoryManagerDump ();
	
	cudaDeviceReset();

	return;
}

void crossbowExecutionContextDump (crossbowExecutionContextP p) {
	int idx;
	int size;
	crossbowDataflowP dataflow;
	size = crossbowArrayListSize (p->dataflows);
	printf ("=== [Execution context: %d dataflows] ===\n", size);
	printf ("=== \n");
        for (idx = 0; idx < size; idx++) {
                dataflow = (crossbowDataflowP) crossbowArrayListGet (p->dataflows, idx);
                crossbowDataflowDumpDependencyGraph (dataflow);
        }
	printf ("=== \n");
	printf ("=== [End of execution context dump] ===\n");
        fflush (stdout);
        return;
}

static unsigned crossbowExecutionContextDeviceIsRequested (int deviceId, int *requests, int numberofrequests) {
	int idx;
	if (numberofrequests == 1 && requests[0] < 0) {
		return 1;
	} else {
		for (idx = 0; idx < numberofrequests; ++idx)
			if (requests[idx] == deviceId)
				return 1;
		return 0;
	}
}

static void crossbowExecutionContextDeviceIsValid (int *requests, int numberofrequests, int limit) {
	int idx;
	for (idx = 0; idx < numberofrequests; ++idx) {
		if ((requests[idx] < 0) || (requests[idx] > (limit - 1))) {
			fprintf(stderr, "error: invalid device request: %d. Valid values in range [0, %d).\n", requests[idx], limit);
			exit(1);
		}
	}
	return;
}

crossbowExecutionContextP crossbowExecutionContextInit
	(int *requests, int numberofrequests, int numberofstreams, int numberofcallbackhandlers, int numberoftaskhandlers, int *offset) {

	int numberofdevices = 0;
	int deviceId = 0;

	crossbowArrayListP devices;
	crossbowDeviceP dev;

	checkCudaErrors (cudaGetDeviceCount (&numberofdevices));
	if (0 == numberofdevices) err("No CUDA device found\n");

	if (numberofrequests > numberofdevices) {
		fprintf(stderr, "error: requested number of devices exceeds available devices\n");
		exit(1);
	}

	/* Devices are number from 0 to N, so make sure that requests are valid */
	crossbowExecutionContextDeviceIsValid (requests, numberofrequests, numberofdevices);

	devices = crossbowArrayListCreate (numberofdevices);
	for (deviceId = 0; deviceId < numberofdevices; ++deviceId)
		crossbowArrayListSet (devices, deviceId, crossbowDeviceCreate (deviceId));

	/* Configure and get GPU device properties */

	struct cudaDeviceProp properties;
	int cpp;

	for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {

		dev = crossbowArrayListGet (devices, deviceId);

		if (crossbowExecutionContextDeviceIsRequested (dev->id, requests, numberofrequests))
			crossbowDeviceSelect (dev);

		/* Get device properties */
		checkCudaErrors(cudaGetDeviceProperties (&properties, dev->id));

		cpp = __convert_sm_to_cores (properties.major, properties.minor);

		info("[%s] %-20s SM %d.%d: %d multiprocessor(s) x %d (cores/multiprocessor), %d cores in total\n",
					crossbowDeviceSelected (dev) ? "*" : " ",
					properties.name,
					properties.major,
					properties.minor,
					properties.multiProcessorCount,
					cpp,
					cpp * properties.multiProcessorCount);

		dev->frequency = ((int64_t) properties.clockRate) * 1000;

		/* Set device flags for mapping host memory */
		if (crossbowDeviceSelected (dev)) {

			if (! properties.canMapHostMemory) err("Cannot map host memory\n");

			checkCudaErrors(cudaSetDevice (dev->id));
			checkCudaErrors(cudaSetDeviceFlags (cudaDeviceScheduleSpin | cudaDeviceMapHost));

			/* Update and check limit of pending kernel launches:
			 *
			 * size_t limit = 32768;
			 * checkCudaErrors(cudaDeviceSetLimit (cudaLimitDevRuntimePendingLaunchCount,  limit));
			 * checkCudaErrors(cudaDeviceGetLimit (&limit, cudaLimitDevRuntimePendingLaunchCount));
			 * info("Maximum number of outstanding runtime launches on device %d is %zu\n", dev->id, limit);
			 */
		}
	}

	crossbowExecutionContextP context = crossbowExecutionContextCreate (devices, numberofstreams, 
		numberofcallbackhandlers, numberoftaskhandlers, offset);

	/* Bind all kernels (function pointers) */
	crossbowExecutionContextBindKernels(context);

	return context;
}

void crossbowExecutionContextSetRandomSeed (crossbowExecutionContextP ctx, unsigned long long seed) {
	int ndx;
	crossbowDeviceP dev;
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		dev = crossbowArrayListGet (ctx->devices, ndx);
		if (crossbowDeviceSelected (dev))
			checkCurandStatus(curandSetPseudoRandomGeneratorSeed(dev->curandGenerator, seed));
	}
	/* Store seed to set curandGenerator per stream */
	ctx->seed = seed;
	return;
}

void crossbowExecutionContextSetKernel (crossbowExecutionContextP ctx, int id, const char *binding, int inputs, int variables, int outputs, int pull) {
	crossbowKernelFunctionP func = crossbowKernelMapResolve (ctx->kernelmap, binding);
	nullPointerException (func);
	dbg ("Register kernel %d (\"%s\", %p)\n", id, binding, func);
	dbg ("%d inputs, %d variables, %d outputs\n", inputs, variables, outputs);
	crossbowKernelP kernel = crossbowKernelCreate (id, binding, func, inputs, variables, outputs, pull);
	crossbowArrayListSet (ctx->kernels, id, kernel);
	return;
}

/* A kernel has one or more inputs, identified by `ndx` */
void crossbowExecutionContextSetKernelInput (crossbowExecutionContextP ctx, int id, int ndx, int argc, int * argv, int capacity) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	crossbowKernelSetInput (kernel, ndx, crossbowVariableSchemaCreate (argc, argv, capacity));
	return;
}

void crossbowExecutionContextSetKernelOutput (crossbowExecutionContextP ctx, int id, int argc, int * argv, int capacity) {
	/*
	 * For each device we create up to `replicas` output buffers to handle
	 * that many pipelined dataflow graph executions
	 */
	crossbowThetaQueueP queue = (crossbowThetaQueueP) crossbowArrayListGet (ctx->streams, ctx->defaultDeviceId);
	int replicas = crossbowThetaQueueSize (queue);
    int splits = crossbowBatchGetSplits (ctx->batch);

	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	crossbowKernelSetOutput (kernel, crossbowVariableSchemaCreate (argc, argv, capacity));
	crossbowKernelSetOutputBufferPool (kernel, replicas * splits, ctx->devices);
	return;
}

void crossbowExecutionContextSetKernelLocalVariable (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, int argc, int *argv, int capacity, int readonly) {

	crossbowThetaQueueP queue = (crossbowThetaQueueP) crossbowArrayListGet (ctx->streams, ctx->defaultDeviceId);
	int replicas = (! readonly) ? crossbowThetaQueueSize (queue) : 0;
    int splits = crossbowBatchGetSplits (ctx->batch);

	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set local variable in `kernel->variables` array list */
	crossbowLocalVariableP p = crossbowArrayListGet (kernel->variables, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s local variable %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowLocalVariableCreate (binding, crossbowVariableCreate (crossbowVariableSchemaCreate (argc, argv, capacity)), readonly, replicas * splits, ctx->devices);
	crossbowArrayListSet (kernel->variables, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelLocalVariableBuffer (crossbowExecutionContextP ctx, int id, int ndx, void *src) {

	int deviceId;
	int numberofdevices;
	crossbowDeviceP dev;

	crossbowDataBufferP data = NULL;
	int offset = 0;
	int length = 0;

	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	crossbowLocalVariableP variable = crossbowArrayListGet (kernel->variables, ndx);
	nullPointerException (variable);
	if (! crossbowLocalVariableReadOnly(variable))
		illegalOperationException ();

	numberofdevices = crossbowArrayListSize(ctx->devices);

	for (deviceId = 0; deviceId < numberofdevices; deviceId++) {
		dev = (crossbowDeviceP) crossbowArrayListGet (ctx->devices, deviceId);
		if (! crossbowDeviceSelected (dev))
			continue;

		/* Assume that this is a read-only local variable, so index to variable does not matter (we set it to -1) */
		invalidConditionException(variable->type == RO);

		data = crossbowLocalVariableGetDataBuffer (variable, dev->id, -1, &offset, &length);
		crossbowDataBufferInitHostRegion (data, offset, src, 0, length);
		/* Copy data to GPU */
		crossbowDataBufferPushSync (data);
	}
	return;
}

void crossbowExecutionContextSetKernelScalars (crossbowExecutionContextP ctx, int id, int count) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	kernel->scalars = crossbowArrayListCreate (count);
	return;
}

void crossbowExecutionContextSetKernelScalarAsInt (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, int value) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set scalar in `kernel->scalars` array list */
	crossbowKernelScalarP p = crossbowArrayListGet (kernel->scalars, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s scalar %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelScalarCreate ();
	crossbowKernelScalarSetIntValue (p, binding, value);
	crossbowArrayListSet (kernel->scalars, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelScalarAsFloat (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, float value) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set scalar in `kernel->scalars` array list */
	crossbowKernelScalarP p = crossbowArrayListGet (kernel->scalars, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s scalar %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelScalarCreate ();
	crossbowKernelScalarSetFloatValue (p, binding, value);
	crossbowArrayListSet (kernel->scalars, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelScalarAsDouble (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, double value) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set scalar in `kernel->scalars` array list */
	crossbowKernelScalarP p = crossbowArrayListGet (kernel->scalars, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s scalar %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelScalarCreate ();
	crossbowKernelScalarSetDoubleValue (p, binding, value);
	crossbowArrayListSet (kernel->scalars, ndx, p);
	return;
}

void crossbowExecutionContextCudnnSetKernelType (crossbowExecutionContextP ctx, int id, int type) {
	int numberofdevices = crossbowArrayListSize (ctx->devices);
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	kernel->cudnnKernelType = type;
	switch (kernel->cudnnKernelType) {
		case CONV:
			kernel->descriptors.conv = crossbowCudnnConvParamsCreate ();
			break;
		case POOL:
			dbg("Creating parameters for kernel %s\n", kernel->name);
			kernel->descriptors.pool = crossbowCudnnPoolParamsCreate ();
			break;
		case RELU:
			kernel->descriptors.relu = crossbowCudnnReLUParamsCreate ();
			break;
		case SOFTMAX:
			dbg("Creating parameters for kernel %s\n", kernel->name);
			kernel->descriptors.softmax = crossbowCudnnSoftMaxParamsCreate ();
			break;
		case BATCHNORM:
			dbg("Creating parameters for kernel %s\n", kernel->name);
			kernel->descriptors.batchnorm = crossbowCudnnBatchNormParamsCreate (numberofdevices);
			break;
		case DROPOUT:
			dbg("Creating parameters for kernel %s\n", kernel->name);
			kernel->descriptors.dropout = crossbowCudnnDropoutParamsCreate (numberofdevices);
			break;
		default:
			err ("error: invalid cuDNN kernel type: %d\n", type);
	}
	return;
}

void crossbowExecutionContextCudnnSetKernelInputDescriptor (crossbowExecutionContextP ctx, int id, int count, int channels, int height, int width) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	switch (kernel->cudnnKernelType) {
		case CONV:
			crossbowCudnnConvParamsSetInputDescriptor (kernel->descriptors.conv, count, channels, height, width);
			break;
		case POOL:
			crossbowCudnnPoolParamsSetInputDescriptor (kernel->descriptors.pool, count, channels, height, width);
			break;
		case RELU:
			crossbowCudnnReLUParamsSetInputDescriptor (kernel->descriptors.relu, count, channels, height, width);
			break;
		case SOFTMAX:
			crossbowCudnnSoftMaxParamsSetInputDescriptor (kernel->descriptors.softmax, count, channels, height, width);
			break;
		case BATCHNORM:
			crossbowCudnnBatchNormParamsSetInputDescriptor (kernel->descriptors.batchnorm, count, channels, height, width);
			break;
		case DROPOUT:
			crossbowCudnnDropoutParamsSetInputDescriptor (kernel->descriptors.dropout, count, channels, height, width);
			break;
		default:
			err ("error: invalid cuDNN kernel type\n");
	}
	return;
}

void crossbowExecutionContextCudnnSetKernelOutputDescriptor (crossbowExecutionContextP ctx, int id, int count, int channels, int height, int width) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	switch (kernel->cudnnKernelType) {
		case CONV:
			crossbowCudnnConvParamsSetOutputDescriptor (kernel->descriptors.conv, count, channels, height, width);
			break;
		case POOL:
			crossbowCudnnPoolParamsSetOutputDescriptor (kernel->descriptors.pool, count, channels, height, width);
			break;
		case RELU:
			crossbowCudnnReLUParamsSetOutputDescriptor (kernel->descriptors.relu, count, channels, height, width);
			break;
		case SOFTMAX:
			crossbowCudnnSoftMaxParamsSetOutputDescriptor (kernel->descriptors.softmax, count, channels, height, width);
			break;
		case BATCHNORM:
			crossbowCudnnBatchNormParamsSetOutputDescriptor (kernel->descriptors.batchnorm, count, channels, height, width);
			break;
		case DROPOUT:
			crossbowCudnnDropoutParamsSetOutputDescriptor (kernel->descriptors.dropout, count, channels, height, width);
			break;
		default:
			err ("error: invalid cuDNN kernel type\n");
	}
	return;
}

void crossbowExecutionContextCudnnSetConvolutionDescriptor (crossbowExecutionContextP ctx, int id, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == CONV);
	crossbowCudnnConvParamsSetConvolutionDescriptor (kernel->descriptors.conv, paddingHeight, paddingWidth, strideHeight, strideWidth);
	return;
}

void crossbowExecutionContextCudnnSetConvolutionFilterDescriptor (crossbowExecutionContextP ctx, int id, int count, int channels, int height, int width) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == CONV);
	crossbowCudnnConvParamsSetFilterDescriptor (kernel->descriptors.conv, count, channels, height, width);
	return;
}

void crossbowExecutionContextCudnnSetConvolutionBiasDescriptor (crossbowExecutionContextP ctx, int id, int count, int channels, int height, int width) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == CONV);
	crossbowCudnnConvParamsSetBiasDescriptor (kernel->descriptors.conv, count, channels, height, width);
	return;
}

void crossbowExecutionContextCudnnSetBatchNormDescriptor (crossbowExecutionContextP ctx, int id) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == BATCHNORM);
	crossbowCudnnBatchNormParamsSetBatchNormDescriptor (kernel->descriptors.batchnorm);
}

void crossbowExecutionContextCudnnSetBatchNormEstimatedMeanAndVariance (crossbowExecutionContextP ctx, int id, int capacity) {
	int ndx;
	crossbowDeviceP dev;
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == BATCHNORM);
	for (ndx = 0; ndx < crossbowArrayListSize (ctx->devices); ++ndx) {
		dev = (crossbowDeviceP) crossbowArrayListGet (ctx->devices, ndx);
		if (crossbowDeviceSelected (dev)) {
			/* Redirect calls to specific device */
			checkCudaErrors (cudaSetDevice(dev->id));
			crossbowCudnnBatchNormParamsSetEstimatedMeanAndVariable (kernel->descriptors.batchnorm, dev->id, capacity, (dev->id == ctx->defaultDeviceId) ? 1 : 0);
		}
	}
}

size_t crossbowExecutionContextCudnnConfigureConvolutionForwardAlgorithm (crossbowExecutionContextP ctx, int id, int limit, double threshold) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	crossbowDeviceP device = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	invalidConditionException(kernel->cudnnKernelType == CONV);
	/* Make sure performance analysis is performed on the default device */
	checkCudaErrors (cudaSetDevice(device->id));
	return crossbowCudnnConvParamsConfigureForward (kernel->descriptors.conv, limit, threshold, device->cudnnHandle);
}

size_t crossbowExecutionContextCudnnConfigureConvolutionBackwardFilterAlgorithm (crossbowExecutionContextP ctx, int id, int limit, double threshold) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	crossbowDeviceP device = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	invalidConditionException(kernel->cudnnKernelType == CONV);
	/* Make sure performance analysis is performed on the default device */
	checkCudaErrors (cudaSetDevice(device->id));
	return crossbowCudnnConvParamsConfigureBackwardFilter (kernel->descriptors.conv, limit, threshold, device->cudnnHandle);
}

size_t crossbowExecutionContextCudnnConfigureConvolutionBackwardDataAlgorithm (crossbowExecutionContextP ctx, int id, int limit, double threshold) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	crossbowDeviceP device = crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	invalidConditionException(kernel->cudnnKernelType == CONV);
	/* Make sure performance analysis is performed on the default device */
	checkCudaErrors (cudaSetDevice(device->id));
	return crossbowCudnnConvParamsConfigureBackwardData (kernel->descriptors.conv, limit, threshold, device->cudnnHandle);
}

void crossbowExecutionContextCudnnSetPoolingMode (crossbowExecutionContextP ctx, int id, int mode) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == POOL);
	crossbowCudnnPoolParamsSetMode (kernel->descriptors.pool, mode);
	return;
}

void crossbowExecutionContextCudnnSetPoolingDescriptor (crossbowExecutionContextP ctx, int id, int windowHeight, int windowWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == POOL);
	crossbowCudnnPoolParamsSetPoolingDescriptor (kernel->descriptors.pool, windowHeight, windowWidth, paddingHeight, paddingWidth, strideHeight, strideWidth);
	return;
}

void crossbowExecutionContextCudnnSetActivationDescriptor (crossbowExecutionContextP ctx, int id, int mode, double ceiling) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == RELU);
	crossbowCudnnReLUParamsSetActivationDescriptor (kernel->descriptors.relu, mode, ceiling);
	return;
}

void crossbowExecutionContextCudnnSetDropoutDescriptor (crossbowExecutionContextP ctx, int id, float dropout, unsigned long long seed) {
	int deviceId;
	crossbowDeviceP dev;
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == DROPOUT);
	int numberofdevices = crossbowArrayListSize (ctx->devices);
	for (deviceId = 0; deviceId < numberofdevices; ++deviceId) {
		dev = (crossbowDeviceP) crossbowArrayListGet (ctx->devices, deviceId);
		if (crossbowDeviceSelected (dev)) {
			checkCudaErrors (cudaSetDevice(dev->id));
			crossbowCudnnDropoutParamsSetDropoutDescriptor (kernel->descriptors.dropout, dev->id, dev->cudnnHandle, dropout, seed);
		}
	}
	return;
}

size_t crossbowExecutionContextCudnnGetDropoutReserveSpaceSize (crossbowExecutionContextP ctx, int id) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	invalidConditionException(kernel->cudnnKernelType == DROPOUT);
	return crossbowCudnnDropoutParamsGetReserveSpaceSize (kernel->descriptors.dropout);
}

void crossbowExecutionContextSetKernelConfigurationParameters (crossbowExecutionContextP ctx, int id, int count) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	kernel->parameters = crossbowArrayListCreate (count);
	return;
}

void crossbowExecutionContextSetKernelConfigurationParameterAsInt (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, int value) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set local variable in `kernel->parameters` array list */
	crossbowKernelConfigParamP p = crossbowArrayListGet (kernel->parameters, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s configuration parameter %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelConfigParamCreate ();
	crossbowKernelConfigParamSetIntValue (p, binding, value);
	crossbowArrayListSet (kernel->parameters, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelConfigurationParameterAsFloat (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, float value) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set local variable in `kernel->parameters` array list */
	crossbowKernelConfigParamP p = crossbowArrayListGet (kernel->parameters, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s configuration parameter %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelConfigParamCreate ();
	crossbowKernelConfigParamSetFloatValue (p, binding, value);
	crossbowArrayListSet (kernel->parameters, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelConfigurationParameterAsIntArray (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, int argc, int *argv) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set local variable in `kernel->parameters` array list */
	crossbowKernelConfigParamP p = crossbowArrayListGet (kernel->parameters, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s configuration parameter %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelConfigParamCreate ();
	crossbowKernelConfigParamSetIntArray (p, binding, argv, argc);
	crossbowArrayListSet (kernel->parameters, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelConfigurationParameterAsFloatArray (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, int argc, float *argv) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set local variable in `kernel->parameters` array list */
	crossbowKernelConfigParamP p = crossbowArrayListGet (kernel->parameters, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s configuration parameter %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelConfigParamCreate ();
	crossbowKernelConfigParamSetFloatArray (p, binding, argv, argc);
	crossbowArrayListSet (kernel->parameters, ndx, p);
	return;
}

void crossbowExecutionContextSetKernelConfigurationParameterAsDouble (crossbowExecutionContextP ctx, int id, int ndx, const char *binding, double value) {
	crossbowKernelP kernel = crossbowArrayListGet (ctx->kernels, id);
	/* Set local variable in `kernel->parameters` array list */
	crossbowKernelConfigParamP p = crossbowArrayListGet (kernel->parameters, ndx);
	if (p) {
		fprintf(stderr, "error: kernel's %s configuration parameter %d already set\n", kernel->name, ndx);
		exit(1);
	}
	p = crossbowKernelConfigParamCreate ();
	crossbowKernelConfigParamSetDoubleValue (p, binding, value);
	crossbowArrayListSet (kernel->parameters, ndx, p);
	return;
}

/* A kernel is referenced by one of more dataflow nodes (or operators) */
void crossbowExecutionContextSetDataflowGraph (crossbowExecutionContextP ctx, int id, int argc, int *argv) {
	int i;
	/*
	 * We create events to mark the beginning and end of each
	 * operator on an execution stream.
	 */
	crossbowThetaQueueP queue = (crossbowThetaQueueP) crossbowArrayListGet (ctx->streams, ctx->defaultDeviceId);
	int events = crossbowThetaQueueSize (queue);
	dbg ("Register dataflow sub-graph %d (%d operators)\n", id, argc);
	crossbowDataflowP dataflow = crossbowDataflowCreate (id);
	for (i = 0; i < argc; ++i) {
		dbg ("Operator's %d kernel id is %d\n", i, argv[i]);
		crossbowKernelP kernel = (crossbowKernelP) crossbowArrayListGet (ctx->kernels, argv[i]);
		nullPointerException (kernel);
		crossbowDataflowAppend (dataflow, kernel, events);
	}
	crossbowArrayListSet (ctx->dataflows, id, dataflow);
	return;
}

void crossbowExecutionContextSetDataflowStream (crossbowExecutionContextP ctx, int id, int ord, int branch) {
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowOperatorP p = crossbowDataflowFindOperator (dataflow, ord);
	nullPointerException(p);
	p->branch = branch;
	invalidConditionException(p->branch >= 0);
	return;
}

void crossbowExecutionContextSetDataflowDependency (crossbowExecutionContextP ctx, int id, int ord, 
	int jtype, int guard, unsigned internal) {
	crossbowOperatorDependency_t type = START_BEFORE_START;
	/* Operator `guard` must start or end before operator `ord` starts. */
	switch (jtype) {
	case 0: type = START_BEFORE_START; break;
	case 1: type =   END_BEFORE_START;
	}
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowOperatorP p = crossbowDataflowFindOperator (dataflow,   ord);
	crossbowOperatorP q = crossbowDataflowFindOperator (dataflow, guard);
	nullPointerException(p);
	nullPointerException(q);
	crossbowOperatorSetTaskDependency (p, type, q, internal);
	q->required ++;
	return;
}

void crossbowExecutionContextSetDataflowUpstreamNeighbours (crossbowExecutionContextP ctx, int id, int ord, int argc, int *argv) {
	int i;
	dbg ("Register upstream neighbours for operator %d:%d (%d operators)\n", id, ord, argc);
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowOperatorP p = crossbowDataflowFindOperator (dataflow, ord);
	nullPointerException(p);
	p->upstream = crossbowArrayListCreate (argc);
	crossbowOperatorP op;
	for (i = 0; i < argc; ++i) {
		op = crossbowDataflowFindOperator (dataflow, argv[i]);
		nullPointerException(op);
		crossbowArrayListSet (p->upstream, i, op);
	}
	return;
}

void crossbowExecutionContextSetDataflowDownstreamNeighbours (crossbowExecutionContextP ctx, int id, int ord, int argc, int *argv) {
	int i;
	dbg ("Register downstream neighbours for operator %d:%d (%d operators)\n", id, ord, argc);
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowOperatorP p = crossbowDataflowFindOperator (dataflow, ord);
	nullPointerException(p);
	p->downstream = crossbowArrayListCreate (argc);
	crossbowOperatorP op;
	for (i = 0; i < argc; ++i) {
		op = crossbowDataflowFindOperator (dataflow, argv[i]);
		nullPointerException(op);
		crossbowArrayListSet (p->downstream, i, op);
	}
	return;
}

void crossbowExecutionContextSetDataflowLossOperator (crossbowExecutionContextP ctx, int id, int op) {
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowDataflowSetLossOperator (dataflow, op);
	return;
}

void crossbowExecutionContextSetDataflowAccuracyOperator (crossbowExecutionContextP ctx, int id, int op) {
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowDataflowSetAccuracyOperator (dataflow, op);
	return;
}

void crossbowExecutionContextSetDataflowDataTransformOperator (crossbowExecutionContextP ctx, int id, int op) {
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowDataflowSetDataTransformOperator (dataflow, op);
	return;
}

/* A kernel is referenced by one of more dataflow nodes (or operators) */
void crossbowExecutionContextSetDataflowPeers (crossbowExecutionContextP ctx, int id, int argc, int *argv) {
	int i;
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	for (i = 0; i < argc; ++i) {
		if (argv[i] < 0)
			continue;
		crossbowOperatorP p = crossbowDataflowFindOperator (dataflow,      i );
		crossbowOperatorP q = crossbowDataflowFindKernel   (dataflow, argv[i]);
		p->peer = q;
	}
	return;
}

void crossbowExecutionContextSetDataflowMemoryPlan (crossbowExecutionContextP ctx, int id, int ord, int provider, int position) {
	if (provider < 0)
		return;
	crossbowDataflowP dataflow = crossbowArrayListGet (ctx->dataflows, id);
	nullPointerException(dataflow);
	crossbowOperatorP p = crossbowDataflowFindOperator (dataflow, ord);
	nullPointerException(p);
	crossbowOperatorP q = crossbowDataflowFindOperator (dataflow, provider);
	nullPointerException(q);
	p->provider = q;
	p->position = position;
	return;
}

void crossbowExecutionContextSetModel (crossbowExecutionContextP ctx, int variables, int size) {
	ctx->theModel = crossbowModelCreate (variables, size, ctx->defaultDeviceId, 1);
	return;
}

void crossbowExecutionContextSetModelVariable (crossbowExecutionContextP ctx, int id, int order, int argc, int *argv, int capacity) {
	crossbowModelRegister (ctx->theModel, id, order, crossbowVariableSchemaCreate (argc, argv, capacity));
	return;
}

void crossbowExecutionContextSetModelVariableBuffer (crossbowExecutionContextP ctx, int id, int order, void *src) {
	crossbowDataBufferP data;
	int offset = 0, length = 0;
	data = crossbowModelVariable (ctx->theModel, id, order, &offset, &length);
	/* Model variable data buffers are always pinned */
	crossbowDataBufferInitHostRegion (data, offset, src, 0, length);
	return;
}

void crossbowExecutionContextSetModelVariableLearningRateMultiplier (crossbowExecutionContextP ctx, int id, int order, float multiplier) {
	crossbowVariableP p = crossbowModelFind (ctx->theModel, id, order);
	nullPointerException(p);
	crossbowVariableSetLearningRateMultiplier (p, multiplier);
	/*
	 * TODO
	 *
	 * An alternative version is to iterate over the model and
	 * assign this variable if any of the multipliers is not 1.
	 */
	if (multiplier != 1)
		ctx->theModel->conf->irregular ++;
	return;
}

void crossbowExecutionContextSetModelWorkPerClock (crossbowExecutionContextP ctx, int wpc) {
	ctx->theModel->wpc = wpc;
	return;
}

void crossbowExecutionContextSetUpdateModelType (crossbowExecutionContextP ctx, int type) {

	if      (type == 0) ctx->theModel->type = DEFAULT;
	else if (type == 1) ctx->theModel->type = WORKER;
	else if (type == 2) ctx->theModel->type = EAMSGD;
	else if (type == 3) ctx->theModel->type = SYNCHRONOUSEAMSGD;
	else if (type == 4) ctx->theModel->type = DOWNPOUR;
	else if (type == 5) ctx->theModel->type = HOGWILD;
	else if (type == 6) ctx->theModel->type = POLYAK_RUPPERT;
	else if (type == 7) ctx->theModel->type = SMA;
	else
		err("Invalid update model type\n");

	return;
}

void crossbowExecutionContextSetEamsgdAlpha (crossbowExecutionContextP ctx, float alpha) {
	ctx->theModel->conf->alpha = alpha;
	return;
}

void crossbowExecutionContextSetEamsgdTau (crossbowExecutionContextP ctx, int tau) {
	ctx->theModel->conf->tau = tau;
	return;
}

void crossbowExecutionContextSetBaseModelMomentum (crossbowExecutionContextP ctx, float baseModelMomentum) {
	ctx->theModel->conf->baseModelMomentum = baseModelMomentum;
	return;
}

void crossbowExecutionContextSetMomentum (crossbowExecutionContextP ctx, float momentum, int method) {
	ctx->theModel->conf->momentum = momentum;
	switch (method) {
	case 0: ctx->theModel->conf->momentumMethod = POLYAK;   break;
	case 1: ctx->theModel->conf->momentumMethod = NESTEROV; break;
	default:
		err("Invalid momentum type\n");
	}
	return;
}

void crossbowExecutionContextSetWeightDecay (crossbowExecutionContextP ctx, float weightDecay) {
	ctx->theModel->conf->weightDecay = weightDecay;
	return;
}

void crossbowExecutionContextSetLearningRateDecayPolicyFixed (crossbowExecutionContextP ctx, float learningRate) {
	ctx->theModel->conf->learningRateDecayPolicy = FIXED;
	ctx->theModel->conf->learningRate = learningRate;
	return;
}

void crossbowExecutionContextSetLearningRateDecayPolicyInv (crossbowExecutionContextP ctx, float learningRate, double gamma, double power) {
	ctx->theModel->conf->learningRateDecayPolicy = INV;
	ctx->theModel->conf->learningRate = learningRate;
	ctx->theModel->conf->gamma = gamma;
	ctx->theModel->conf->power = power;
	return;
}

void crossbowExecutionContextSetLearningRateDecayPolicyStep (crossbowExecutionContextP ctx, float learningRate, double gamma, int size) {
	ctx->theModel->conf->learningRateDecayPolicy = STEP;
	ctx->theModel->conf->learningRate = learningRate;
	ctx->theModel->conf->gamma = gamma;
	ctx->theModel->conf->size = size;
	return;
}

void crossbowExecutionContextSetLearningRateDecayPolicyMultiStep (crossbowExecutionContextP ctx, float learningRate, double gamma, int warmuptasks, int argc, int* argv) {
	int i;
	ctx->theModel->conf->learningRateDecayPolicy = ((warmuptasks > 0) ? LSR : MULTISTEP);
	ctx->theModel->conf->learningRate = learningRate;
	ctx->theModel->conf->gamma = gamma;
	ctx->theModel->conf->warmuptasks = warmuptasks;
	ctx->theModel->conf->numberofsteps = argc;
	ctx->theModel->conf->steps = crossbowMalloc (argc * sizeof(int));
	for (i = 0; i < argc; ++i)
		ctx->theModel->conf->steps[i] = argv[i];
	return;
}

void crossbowExecutionContextSetLearningRateDecayPolicyExp (crossbowExecutionContextP ctx, float learningRate, double gamma) {
	ctx->theModel->conf->learningRateDecayPolicy = EXP;
	ctx->theModel->conf->learningRate = learningRate;
	ctx->theModel->conf->gamma = gamma;
	return;
}

void crossbowExecutionContextSetLearningRateDecayPolicyCircular (crossbowExecutionContextP ctx, float *circularLearningRate, int superConvergence, float *circularMomentum, int size) {
	int i;

	ctx->theModel->conf->learningRateDecayPolicy = CLR;

	ctx->theModel->conf->superConvergence = superConvergence;

	ctx->theModel->conf->circularLearningRate = (float *) crossbowMalloc (3 * sizeof(float));
	ctx->theModel->conf->circularMomentum     = (float *) crossbowMalloc (3 * sizeof(float));

	for (i = 0; i < 3; ++i) {
		ctx->theModel->conf->circularLearningRate [i] = circularLearningRate [i];
		ctx->theModel->conf->circularMomentum     [i] = circularMomentum     [i];
	}

	ctx->theModel->conf->size = size;
	return;
}

void crossbowExecutionContextSetModelManager (JNIEnv *env, crossbowExecutionContextP ctx, int replicas, int type) {
	/* Get number of streams on default device */
	int streams = crossbowThetaQueueSize
			((crossbowThetaQueueP) crossbowArrayListGet(ctx->streams, ctx->defaultDeviceId));
	if (streams != replicas)
		warn("Number of replicas is not equal to the number of streams\n");
#ifdef GPU_VERBOSE
	crossbowModelDump (ctx->theModel);
#endif
	/* Find synchronisation type */
	crossbowModelSynchronisation_t syncType;
	if      (type == 0) syncType = BSP;
	else if (type == 1) syncType = SSP;
	else if (type == 2) syncType = ASP;
	else
		illegalStateException();

	/* Finalise the model */
	crossbowModelFinalise(ctx->theModel);

	/* Push the model's initialised variables to GPU memory */
	checkCudaErrors (cudaSetDevice (ctx->theModel->dev));
	crossbowDataBufferPushSync (ctx->theModel->data);

	/* Create model manager */
	ctx->modelmanager = crossbowModelManagerCreate (env, replicas, ctx->theModel, syncType, ctx->devices);

	/* Set the model manager to all the callback handler (their pointers are null). */
	int socketId, handlerId = 0;
	crossbowCallbackHandlerP handler;

	for (socketId = 0; socketId < crossbowArrayListSize(ctx->callbackhandlers); ++socketId) {

		/* Get pool of handlers for a particular socket */
		crossbowArrayListP pool = (crossbowArrayListP) crossbowArrayListGet (ctx->callbackhandlers, socketId);

        for (handlerId = 0; handlerId < crossbowArrayListSize(pool); ++handlerId) {

            handler = crossbowArrayListGet (pool, handlerId);
		    handler->modelmanager = ctx->modelmanager;
        }
	}

	return;
}

jobject crossbowExecutionContextAcquireAccess (JNIEnv *env, crossbowExecutionContextP ctx, int *argv) {
	jobject result = NULL;
	result = crossbowModelManagerAcquireAccess (env, ctx->modelmanager, &argv[0]);
	return result;
}

jobject crossbowExecutionContextUpgradeAccess (JNIEnv *env, crossbowExecutionContextP ctx, jobject replicaId, int *argv) {
	jobject result = NULL;
	result = crossbowModelManagerUpgradeAccess (env, ctx->modelmanager, replicaId, &argv[0]);
	return result;
}

void crossbowExecutionContextSetResultHandler (crossbowExecutionContextP ctx, int id, void *slots, int count) {
	crossbowResultHandlerSet (ctx->resulthandler, id, slots, count);
	return;
}

void crossbowExecutionContextSetLightWeightDatasetHandler (crossbowExecutionContextP ctx, int id, void *slots, int count) {
    if (id == 0) {
        /* When we configure the first dataset, we create the handler */
	    invalidConditionException (ctx->datasethandler == NULL);
	    ctx->datasethandler = crossbowLightWeightDatasetHandlerCreate (64); /* Slots are 64-byte wide */
    } else {
        /* Subsequent datasets need only to configure their corresponding slots */
        invalidConditionException (ctx->datasethandler != NULL);
    }
	crossbowLightWeightDatasetHandlerSet (ctx->datasethandler, id, slots, count);
	return;
}

void crossbowExecutionContextExecute (
	JNIEnv *env,
	crossbowExecutionContextP ctx,
	int dataflowId,
	int taskId,
	void *examplesP, int examplesCapacity, int examplesStartP, int examplesEndP,
	void *labelsP,   int   labelsCapacity, int   labelsStartP, int   labelsEndP,
	long *argv,
	int phase,
	jobject replica) {

	dbg("Execute task %04d (dataflow %d phase %d): examples <%p [%10d, %10d) free %10ld> labels <%p [%10d, %10d) free %10ld>\n",
			taskId,
			dataflowId,
			phase,
			examplesP, examplesStartP, examplesEndP, argv[0],
			  labelsP,   labelsStartP,   labelsEndP, argv[1]);

	crossbowModelP model = crossbowModelManagerGet (env, ctx->modelmanager, replica);

	/* Get device on which the model replica resides */
	crossbowDeviceP dev = crossbowArrayListGet (ctx->devices, model->dev);

	/* Find next available stream */
	crossbowStreamP stream = crossbowExecutionContextNextStream (ctx, dev->id);

	/* info("Execute task on device %d stream %d with model replica %2d\n", dev->id, stream->id, model->id); */

	stream->task = taskId;
	stream->phi  =  phase;

	stream->freeP[0] = argv[0];
	stream->freeP[1] = argv[1];

	stream->dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, dataflowId);

	stream->model = model;
	
	if (stream->theModel == NULL)
		stream->theModel = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

	if (stream->modelSynchronisationHandle == NULL)
		stream->modelSynchronisationHandle = dev->modelSynchronisationHandle;

	if (stream->modelSynchronisationStream == NULL)
		stream->modelSynchronisationStream = dev->modelSynchronisationStream;
	
	checkCudaErrors (cudaSetDevice(dev->id));

	/* Initialise stream input, case __INPUT_ISPINNED_:
	 *
	 * Any variable points to the data buffer that holds its data, both on host and on device.
	 * Since the input variables are pinned, input data has to be copied to the corresponding
	 * pinned memory region.
	 *
	 * else:
	 *
	 * Input variables hold a pointer to the input data (e.g. `examplesP`) which must be page
	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
	 *
	 */
#ifndef __INPUT_IS_PINNED_
	if (ctx->datasethandler) {
		/*
		 * Pointers to example and label data are already registered, but we have to make sure
		 * that the data has been copied from the data set files to those buffers.
		 */
		crossbowLightWeightDatasetHandlerReady (ctx->datasethandler, phase, crossbowLightWeightDatasetHandlerTranslate (argv[0], examplesEndP - examplesStartP));
    }
	crossbowHostRegisterBuffer (0, examplesP, examplesCapacity, examplesStartP, examplesEndP, phase);
	crossbowHostRegisterBuffer (1, labelsP,   labelsCapacity,   labelsStartP,   labelsEndP,   phase);
#else
	(void) examplesCapacity;
	(void) labelsCapacity
#endif
	crossbowVariableSetHostData (stream->examples, examplesP, examplesStartP, examplesEndP);
	crossbowVariableSetHostData (stream->labels,   labelsP,   labelsStartP,   labelsEndP  );

#ifndef USE_TASKHANDLERS
	crossbowStreamExecute (ctx, stream);
#else
	crossbowArrayListP pool = crossbowArrayListGet(ctx->taskhandlers, (model->dev < 4) ? 0 : 1);
	crossbowTaskHandlerP taskhandler = crossbowArrayListGetNext (pool);
	dbg("Assign task %04d to task handler #%lu\n", stream->task, taskhandler->id);
	crossbowTaskHandlerPublish (taskhandler, stream);
#endif
	return;
}

void crossbowExecutionContextExecuteNext (
	JNIEnv *env,
	crossbowExecutionContextP ctx,
	int dataflowId,
	int taskId,
	int examplesStartP, int examplesEndP,
	int   labelsStartP, int   labelsEndP,
	long *argv,
	int phase,
	jobject replica) {
	
	dbg("Execute task %04d (dataflow %d phase %d): examples <%p [%10d, %10d) free %10ld> labels <%p [%10d, %10d) free %10ld>\n",
			taskId,
			dataflowId,
			phase,
			NULL, examplesStartP, examplesEndP, argv[0],
			NULL,   labelsStartP,   labelsEndP, argv[1]);
		
	if ((examplesStartP == 0) && (labelsStartP == 0)) {
		/*
		 * N tasks has been scheduled, and the next N tasks
		 * are about to be scheduled. Wait until the images
		 * and labels have been decoded.
		 *
		 * However, what if the swap succeeds but one of the
		 * previous N tasks is still being scheduled?
		 */
		crossbowRecordDatasetSwap (ctx->dataset[phase]);
	}

	void *examplesP = ctx->dataset[phase]->images;
	void   *labelsP = ctx->dataset[phase]->labels;

	nullPointerException (examplesP);
	nullPointerException (  labelsP);

	int examplesCapacity = ctx->dataset[phase]->buffer->capacity[0];
	int   labelsCapacity = ctx->dataset[phase]->buffer->capacity[1];

	crossbowModelP model = crossbowModelManagerGet (env, ctx->modelmanager, replica);

	/* Get device on which the model replica resides */
	crossbowDeviceP dev = crossbowArrayListGet (ctx->devices, model->dev);

	/* Find next available stream */
	crossbowStreamP stream = crossbowExecutionContextNextStream (ctx, dev->id);

	/* info("Execute task on device %d stream %d with model replica %2d\n", dev->id, stream->id, model->id); */

	stream->task = taskId;
	stream->phi  =  phase;

	stream->freeP[0] = argv[0];
	stream->freeP[1] = argv[1];

	stream->dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, dataflowId);

	stream->model = model;

	if (stream->theModel == NULL)
		stream->theModel = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

	if (stream->modelSynchronisationHandle == NULL)
		stream->modelSynchronisationHandle = dev->modelSynchronisationHandle;

	if (stream->modelSynchronisationStream == NULL)
		stream->modelSynchronisationStream = dev->modelSynchronisationStream;
	
	/* Assign dataset to stream */
        stream->dataset = ctx->dataset[phase];
	
	checkCudaErrors (cudaSetDevice(dev->id));

	/* Initialise stream input, case __INPUT_ISPINNED_:
	 *
	 * Any variable points to the data buffer that holds its data, both on host and on device.
	 * Since the input variables are pinned, input data has to be copied to the corresponding
	 * pinned memory region.
	 *
	 * else:
	 *
	 * Input variables hold a pointer to the input data (e.g. `examplesP`) which must be page
	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
	 *
	 */
#ifndef __INPUT_IS_PINNED_
	if (ctx->datasethandler) {
		/*
		 * Pointers to example and label data are already registered, but we have to make sure
		 * that the data has been copied from the data set files to those buffers.
		 */
		crossbowLightWeightDatasetHandlerReady (ctx->datasethandler, phase, crossbowLightWeightDatasetHandlerTranslate (argv[0], examplesEndP - examplesStartP));
    }
	crossbowHostRegisterBuffer (0, examplesP, examplesCapacity, examplesStartP, examplesEndP, phase);
	crossbowHostRegisterBuffer (1, labelsP,   labelsCapacity,   labelsStartP,   labelsEndP,   phase);
#else
	(void) examplesCapacity;
	(void) labelsCapacity
#endif
	crossbowVariableSetHostData (stream->examples, examplesP, examplesStartP, examplesEndP);
	crossbowVariableSetHostData (stream->labels,   labelsP,   labelsStartP,   labelsEndP  );

#ifndef USE_TASKHANDLERS
	crossbowStreamExecute (ctx, stream);
#else
	crossbowArrayListP pool = crossbowArrayListGet(ctx->taskhandlers, (model->dev < 4) ? 0 : 1);
	crossbowTaskHandlerP taskhandler = crossbowArrayListGetNext (pool);
	dbg("Assign task %04d to task handler #%lu\n", stream->task, taskhandler->id);
	crossbowTaskHandlerPublish (taskhandler, stream);
#endif
	return;
}

void crossbowExecutionContextSchedule (
	JNIEnv *env,
	crossbowExecutionContextP ctx,
	int dataflowId,
	int taskId,
	void *examplesP, int examplesCapacity, int examplesStartP, int examplesEndP,
	void *labelsP,   int   labelsCapacity, int   labelsStartP, int   labelsEndP,
	long *argv,
	int phase,
	int bound) {

	(void) env;

	dbg("Schedule task %04d (dataflow %d phase %d): examples <%p [%10d, %10d) free %10ld> labels <%p [%10d, %10d) free %10ld>\n",
			taskId,
			dataflowId,
			phase,
			examplesP, examplesStartP, examplesEndP, argv[0],
			  labelsP,   labelsStartP,   labelsEndP, argv[1]);

	crossbowModelP model = crossbowModelManagerGetNextOrWait (env, ctx->modelmanager, bound);

	/* Get device on which the model replica resides */
	crossbowDeviceP dev = crossbowArrayListGet (ctx->devices, model->dev);

	/* Find next available stream */
	crossbowStreamP stream = crossbowExecutionContextNextStream (ctx, dev->id);

	// info("Execute task on device %d stream %d with model replica %2d\n", dev->id, stream->id, model->id);

	stream->task = taskId;
	stream->phi  =  phase;

	stream->freeP[0] = argv[0];
	stream->freeP[1] = argv[1];

	stream->dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, dataflowId);

	stream->model = model;

	if (stream->theModel == NULL)
		stream->theModel = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

	if (stream->modelSynchronisationHandle == NULL)
		stream->modelSynchronisationHandle = dev->modelSynchronisationHandle;

	if (stream->modelSynchronisationStream == NULL)
		stream->modelSynchronisationStream = dev->modelSynchronisationStream;

	checkCudaErrors (cudaSetDevice(dev->id));

	/* Initialise stream input, case __INPUT_ISPINNED_:
	 *
	 * Any variable points to the data buffer that holds its data, both on host and on device.
	 * Since the input variables are pinned, input data has to be copied to the corresponding
	 * pinned memory region.
	 *
	 * else:
	 *
	 * Input variables hold a pointer to the input data (e.g. `examplesP`) which must be page
	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
	 *
	 */
#ifndef __INPUT_IS_PINNED_
	if (ctx->datasethandler) {
		/*
		 * Pointers to example and label data are already registered, but we have to make sure
		 * that the data has been copied from the data set files to those buffers.
		 */
		crossbowLightWeightDatasetHandlerReady (ctx->datasethandler, phase, crossbowLightWeightDatasetHandlerTranslate (argv[0], examplesEndP - examplesStartP));
    }
	crossbowHostRegisterBuffer (0, examplesP, examplesCapacity, examplesStartP, examplesEndP, phase);
	crossbowHostRegisterBuffer (1, labelsP,   labelsCapacity,   labelsStartP,   labelsEndP,   phase);
#else
	(void) examplesCapacity;
	(void) labelsCapacity
#endif
	crossbowVariableSetHostData (stream->examples, examplesP, examplesStartP, examplesEndP);
	crossbowVariableSetHostData (stream->labels,   labelsP,   labelsStartP,   labelsEndP  );

#ifndef USE_TASKHANDLERS
	crossbowStreamExecute (ctx, stream);
#else
    crossbowArrayListP pool = crossbowArrayListGet(ctx->taskhandlers, (model->dev < 4) ? 0 : 1);
	crossbowTaskHandlerP taskhandler = crossbowArrayListGetNext (pool);
	dbg("Assign task %04d to task handler #%lu\n", stream->task, taskhandler->id);
	crossbowTaskHandlerPublish (taskhandler, stream);
#endif
	return;
}

void crossbowExecutionContextScheduleNext (
	JNIEnv *env,
	crossbowExecutionContextP ctx,
	int dataflowId,
	int taskId,
	int examplesStartP, int examplesEndP,
	int   labelsStartP, int   labelsEndP,
	long *argv,
	int phase,
	int bound) {

	(void) env;

	dbg("Schedule task %04d (dataflow %d phase %d): examples <%p [%10d, %10d) free %10ld> labels <%p [%10d, %10d) free %10ld>\n",
			taskId,
			dataflowId,
			phase,
			NULL, examplesStartP, examplesEndP, argv[0],
			NULL,   labelsStartP,   labelsEndP, argv[1]);

	if ((examplesStartP == 0) && (labelsStartP == 0)) {
		/* 
		 * N tasks has been scheduled, and the next N tasks 
		 * are about to be scheduled. Wait until the images
		 * and labels have been decoded.
		 *
		 * However, what if the swap succeeds but one of the 
		 * previous N tasks is still being scheduled?
		 */
		crossbowRecordDatasetSwap (ctx->dataset[phase]);
	}
	
	void *examplesP = ctx->dataset[phase]->images;
	void   *labelsP = ctx->dataset[phase]->labels;

	nullPointerException (examplesP);
	nullPointerException (  labelsP);

	int examplesCapacity = ctx->dataset[phase]->buffer->capacity[0];
	int   labelsCapacity = ctx->dataset[phase]->buffer->capacity[1];

	crossbowModelP model = crossbowModelManagerGetNextOrWait (env, ctx->modelmanager, bound);

	/* Get device on which the model replica resides */
	crossbowDeviceP dev = crossbowArrayListGet (ctx->devices, model->dev);

	/* Find next available stream */
	crossbowStreamP stream = crossbowExecutionContextNextStream (ctx, dev->id);

	/* info("Execute task on device %d stream %d with model replica %2d\n", dev->id, stream->id, model->id); */

	stream->task = taskId;
	stream->phi  =  phase;

	stream->freeP[0] = argv[0];
	stream->freeP[1] = argv[1];

	stream->dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, dataflowId);

	stream->model = model;

	if (stream->theModel == NULL)
		stream->theModel = crossbowArrayListGet (ctx->modelmanager->baseModels, dev->id);

	if (stream->modelSynchronisationHandle == NULL)
		stream->modelSynchronisationHandle = dev->modelSynchronisationHandle;

	if (stream->modelSynchronisationStream == NULL)
		stream->modelSynchronisationStream = dev->modelSynchronisationStream;
	
	/* Assign dataset to stream */
	stream->dataset = ctx->dataset[phase];

	checkCudaErrors (cudaSetDevice(dev->id));

	/* Initialise stream input, case __INPUT_ISPINNED_:
	 *
	 * Any variable points to the data buffer that holds its data, both on host and on device.
	 * Since the input variables are pinned, input data has to be copied to the corresponding
	 * pinned memory region.
	 *
	 * else:
	 *
	 * Input variables hold a pointer to the input data (e.g. `examplesP`) which must be page
	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
	 *
	 */
#ifndef __INPUT_IS_PINNED_
	crossbowHostRegisterBuffer (0, examplesP, examplesCapacity, examplesStartP, examplesEndP, phase);
	crossbowHostRegisterBuffer (1, labelsP,   labelsCapacity,   labelsStartP,   labelsEndP,   phase);
#else
	(void) examplesCapacity;
	(void) labelsCapacity
#endif
	crossbowVariableSetHostData (stream->examples, examplesP, examplesStartP, examplesEndP);
	crossbowVariableSetHostData (stream->labels,   labelsP,   labelsStartP,   labelsEndP  );

#ifndef USE_TASKHANDLERS
	crossbowStreamExecute (ctx, stream);
#else
    crossbowArrayListP pool = crossbowArrayListGet(ctx->taskhandlers, (model->dev < 4) ? 0 : 1);
	crossbowTaskHandlerP taskhandler = crossbowArrayListGetNext (pool);
	dbg("Assign task %04d to task handler #%lu\n", stream->task, taskhandler->id);
	crossbowTaskHandlerPublish (taskhandler, stream);
#endif
	return;
}

int crossbowExecutionContextLockModels (crossbowExecutionContextP ctx) {
	switch (ctx->modelmanager->type) {
		case BSP:
			if (! crossbowModelManagerLockAll(ctx->modelmanager)) {
				fprintf(stderr, "error: failed to lock all GPU model replicas at synchronisation barrier\n");
				exit(1);
			}
			return ctx->modelmanager->size;
		case SSP:
		case ASP:
			return crossbowModelManagerLockAny(ctx->modelmanager);
		default:
			illegalStateException();
	}
}

/*
 * The function returns the model replica id whose CPU buffer incorporates
 * the most recent update(s).
 *
 * It also returns the number of updates incorporated into the model.
 */
int crossbowExecutionContextMergeModels (crossbowExecutionContextP ctx, int pull) {

	(void) pull;

	int first;

	int N = crossbowModelManagerNumberOfVisibleUpdates (ctx->modelmanager);
	dbg("%d visible updates\n", N);
	/* Return number of updates */
	invalidConditionException(N >= 0);
	/* There should be at least one model update */
	if (N == 0)
		return -1;

	/* If there is a single model, no need to continue */
	if (ctx->modelmanager->size == 1)
		return 0;

	/* Find the first locked model */
	for (first = 0; first < ctx->modelmanager->size; ++first) {
		if (ctx->modelmanager->locked[first])
			break;
	}

	return first;
}

static void crossbowExecutionContextSynchroniseBatchNorm (crossbowExecutionContextP ctx) {
	crossbowOperatorP op;
	crossbowDataflowP dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, 0);
	crossbowDeviceP defaultDev = (crossbowDeviceP) crossbowArrayListGet (ctx->devices, ctx->defaultDeviceId);
	op = crossbowDataflowPeek (dataflow);
	while (op != dataflow->tail) {
		/* Check operator kernel type */
		if (op->kernel->cudnnKernelType == BATCHNORM) {
			crossbowCudnnBatchNormParamsSynchroniseEstimatedMeanAndVariable
				(op->kernel->descriptors.batchnorm, defaultDev);
		}
		op = op->next; /* Get next operator */
	}
	return;
}

int crossbowExecutionContextSynchroniseModels (crossbowExecutionContextP ctx, int first, int clock, int autotune, int push) {

	(void) push;

	dbg("Synchronise GPU models at clock %d\n", clock);
	
	/* crossbowExecutionContextSynchroniseBatchNorm (ctx); */

	switch (ctx->modelmanager->theModel->type) {

	case DEFAULT:
		dbg("Synchronise replicas using default model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationDefault (ctx, first, clock);
		break;

	case WORKER:
		dbg("Synchronise replicas using S-SGD model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationSynchronousSGD (ctx, first, clock);
		break;

	case EAMSGD:
		dbg("Synchronise replicas using EAMSGD model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationElasticAveragingSGD (ctx, first, clock);
		break;

	case SYNCHRONOUSEAMSGD:
		dbg("Synchronise replicas using synchronous EAMSGD model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		/* This ifdef should be removed */
#ifdef ELASTIC_AVERAGE
		crossbowSynchronisationSynchronousElasticAveragingSGD (ctx, first, clock);
#else
		crossbowSynchronisationSMA (ctx, first, clock);
#endif
		break;

	case DOWNPOUR:
		dbg("Synchronise replicas using DOWNPOUR model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationDownpour (ctx, first, clock);
		break;
		
	case HOGWILD:
		dbg("Synchronise replicas using Hogwild! model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationHogwild (ctx, first, clock);
		break;
	
	case POLYAK_RUPPERT:
		dbg("Synchronise replicas using Polyak-Ruppert model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationPolyakRuppert (ctx, first, clock);
		break;
		
	case SMA:
		dbg("Synchronise replicas using SMA model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
		crossbowSynchronisationSMA (ctx, first, clock);
		break;

	default:
		err("Invalid model update type\n");
	}

	if (autotune < 0) {
		crossbowModelManagerDelModel (ctx->modelmanager);
	}
	
	if (autotune > 0) {
		crossbowModelManagerAddModel (ctx->modelmanager);
		crossbowExecutionContextAddStream (ctx);
	}
	
	/* Increment the clock of all locked models */
	crossbowModelManagerIncrementClock(ctx->modelmanager, clock);

	return 0;
}

int crossbowExecutionContextUnlockModels(crossbowExecutionContextP ctx) {
	return crossbowModelManagerUnlockAny(ctx->modelmanager);
}

int crossbowExecutionContextCheckpointModels (crossbowExecutionContextP ctx, const char *dir) {
	int result;
	crossbowOperatorP op;
	crossbowDataflowP dataflow;
	char *pathname = crossbowStringConcat ("%s/%06llu", dir, ++(ctx->version));
	/* Create directory */
	result = mkdir (pathname, 0777);
	if (result < 0)
		err("Failed to create directory %s\n", pathname);
	/* Checkpoint models */
	info("Checkpoint models to %s\n", pathname);
	crossbowModelManagerStore (ctx->modelmanager, pathname);
	/* Checkpoint batch normalisation state, if any such operators are present */
	info("Checkpoint BatchNorm operator state\n");
	dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, 0);
	op = crossbowDataflowPeek (dataflow);
	while (op != dataflow->tail) {
		/* Check operator kernel type */
		if (op->kernel->cudnnKernelType == BATCHNORM) {
			/* Store */
			crossbowCudnnBatchNormParamsStoreEstimatedMeanAndVariable 
					(op->kernel->descriptors.batchnorm, pathname, op->id);
		}
		op = op->next; /* Get next operator */
	}
	crossbowStringFree (pathname);
	return 0;
}

int crossbowExecutionContextOverrideModelData (crossbowExecutionContextP ctx, const char *dir) {
	crossbowOperatorP op;
	crossbowDataflowP dataflow;
	info("Override models from %s\n", dir);
	crossbowModelManagerLoad (ctx->modelmanager, dir);
	/* Override batch normalisation state, if any such operators are present */
	info("Override BatchNorm operator state\n");
	dataflow = (crossbowDataflowP) crossbowArrayListGet (ctx->dataflows, 0);
	op = crossbowDataflowPeek (dataflow);
	while (op != dataflow->tail) {
		/* Check operator kernel type */
		if (op->kernel->cudnnKernelType == BATCHNORM) {
			/* Load */
			crossbowCudnnBatchNormParamsLoadEstimatedMeanAndVariable 
					(op->kernel->descriptors.batchnorm, dir, op->id);
		}
		op = op->next; /* Get next operator */
	}
	return 0;
}

int crossbowExecutionContextAddModel (crossbowExecutionContextP ctx) {
	crossbowModelManagerAddModel (ctx->modelmanager);
	return 0;
}

int crossbowExecutionContextDelModel (crossbowExecutionContextP ctx) {
	crossbowModelManagerDelModel (ctx->modelmanager);
	return 0;
}

/* Record dataset */

void crossbowExecutionContextRecordDatasetInit (crossbowExecutionContextP ctx, int phase, int workers, int *capacity, int NB, int b, int *padding) {
	invalidConditionException ((! ctx->dataset[phase]));
	ctx->dataset[phase] = crossbowRecordDatasetCreate (workers, capacity, NB, b, padding, phase);
	/* 
	 * Let's initialise the random number generator here, 
	 * although it will be called twice.
	 */
	crossbowYarngInit (123456789);
	return;
}

/* Record dataset: populate dataset (with a file) */

void crossbowExecutionContextRecordDatasetRegister (crossbowExecutionContextP ctx, int phase, int id, const char *filename) {
	(void) id;
	dbg("Register record file %d:%03d: %s\n", phase, id, filename);
	crossbowRecordReaderRegister (ctx->dataset[phase]->reader, filename);
	return;
}

/* Record dataset: finalise */

void crossbowExecutionContextRecordDatasetFinalise (crossbowExecutionContextP ctx, int phase) {

	info("Finalise record reader\n");
	crossbowRecordReaderFinalise (ctx->dataset[phase]->reader);
	/* Fill the first buffer */
	info("Fill buffer for the first time (phase: %d)\n", phase);
	crossbowRecordDatasetInitSafely (ctx->dataset[phase]);
	return;
}
