package uk.ac.imperial.lsds.crossbow.device;

import java.nio.ByteBuffer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.task.Task;
import uk.ac.imperial.lsds.crossbow.types.HandlerType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class TheGPU {
	
	private final static Logger log = LogManager.getLogger (TheGPU.class);
	
	private static final TheGPU gpuInstance = new TheGPU ();
	
	public static TheGPU getInstance () { return gpuInstance; }
	
	private boolean loaded;
	
	public TheGPU () {
		loaded = false;
	}
	
	public boolean isLoaded () {
		return loaded;
	}
	
	public void init () {
		
		if (! SystemConf.getInstance().getGPU())
			return;
		
		if (! loaded) {
			String library = String.format("%s/clib-multigpu/libGPU.so", SystemConf.getInstance().getHomeDirectory());
			try {
				System.load (library);
			} catch (final UnsatisfiedLinkError e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			loaded = true;
		}
		
		init (
			SystemConf.getInstance().getGPUDevices(), 
			SystemConf.getInstance().numberOfGPUStreams(), 
			SystemConf.getInstance().numberOfGPUCallbackHandlers(),
			SystemConf.getInstance().numberOfGPUTaskHandlers(),
			SystemConf.getInstance().getCoreMapper().getOffset(HandlerType.CALLBACK), 
			SystemConf.getInstance().getCoreMapper().getOffset(HandlerType.TASK)
			);
		/* Set random seed for cuRAND */
		setRandomSeed (SystemConf.getInstance().getRandomSeed());
	}
	
	public void destroy () {
		if (! SystemConf.getInstance().getGPU())
			return;
		free ();
	}
	
	public void register (ExecutionContext context) {
		
		if (! SystemConf.getInstance().getGPU())
			return;
		
		if (! isLoaded())
			throw new IllegalStateException ("error: GPU library is not loaded");
		
		Dataflow [] dataflows = context.getDataflows();
		
		int branchFactor = 0;
		
		for (int i = 0; i < dataflows.length; ++i) {
			
			Dataflow dataflow = dataflows[i];
			
			if (dataflow == null)
				continue;
			
			SubGraph next = dataflow.getSubGraph();
			while (next != null) {
				
				DataflowNode node = next.getDataflowNode();
				while (node != null) {
					
					node.getOperator().GPURegister();
					node = node.getNextInTopology();
				}
				
				next.GPURegister();
				
				if (branchFactor < next.getBranchFactor())
					branchFactor = next.getBranchFactor();
				
				next = next.getNext();
			}
		}
		
		log.debug(String.format("%d unique operators registered in %d sub-graphs", Operator.cardinality(), SubGraph.cardinality()));
		
		/* Register the model */
		context.getModel().GPURegister();
		context.getModelManager().GPURegister();
		
		TheGPU.getInstance().overrideModelData (SystemConf.getInstance().getModelDirectory ());
		
		/* Configure the input batch */
		Shape inputShape = ModelConf.getInstance().getInputShape (Phase.TRAIN);
		int [] taskSize = ModelConf.getInstance().getTaskSize();
		
		configureBatchExamples (inputShape.array(), taskSize[0]);
		configureBatchLabels (new int [] { inputShape.get(0) }, taskSize[1]);
		configureBatchSplits (ModelConf.getInstance().numberOfSplits());
		
		configureStreams (branchFactor);
		
		if (log.isDebugEnabled())
			dump ();
	}
	
	public void setModelVariableData (int ndx, int order, IDataBuffer buffer) {
		
		if (buffer.isDirect())
			setModelVariableBuffer (ndx, order, buffer.getByteBuffer());
		else
			throw new UnsupportedOperationException ("error: GPU does not support indirect byte buffers yet");
	}
	
	public void setKernelLocalVariableData (int id, int ndx, IDataBuffer buffer) {
		
		if (buffer.isDirect())
			setKernelLocalVariableBuffer (id, ndx, buffer.getByteBuffer());
		else
			throw new UnsupportedOperationException ("error: GPU does not support indirect byte buffers yet");
	}
	
	public void execute (int dataflowId, Batch batch, Integer replicaId, Task task) {
		
		/*
		 * Examples and labels are now special mapped data buffers
		 * and they don't have an underlying byte buffer.
		 */
		IDataBuffer examples = batch.getInputBuffer(0);
		IDataBuffer labels   = batch.getInputBuffer(1);
		
		int [] from = batch.getBufferStartPointers();
		int [] to = batch.getBufferEndPointers();
		
		long [] free = batch.getFreeOffsets();
		
		int phase = (task.isValidationTask()) ? Phase.CHECK.getId() : Phase.TRAIN.getId();
		
		execute (dataflowId, batch.getId(), examples, from[0], to[0], labels, from[1], to[1], free, phase, replicaId);
	}
	
	private native int init (int [] devices, int streams, int callbackhandlers, int taskhandlers, int callbackhandlercoreoffset, int taskhandlercoreoffset);
	private native int free ();
	
	private native int dump ();
	
	private native int configureBatchExamples (int [] shape, int bytes);
	private native int configureBatchLabels (int [] shape, int bytes);
	private native int configureBatchSplits (int splits);
	
	private native int configureStreams (int branches);
	
	public native int setRandomSeed (long seed);
	
	public native int setKernel (int id, String name, int inputs, int variables, int outputs, boolean pull);
	
	public native int setKernelInput  (int id, int ndx, int [] shape, int capacity);
	public native int setKernelOutput (int id, int [] shape, int capacity);
	
	public native int setKernelLocalVariable (int id, int ndx, String name, int [] shape, int capacity, boolean readonly);
	
	/* Initialises read-only local variables */
	public native int setKernelLocalVariableBuffer (int id, int ndx, ByteBuffer buffer);
	
	public native int setKernelConfigurationParameters (int id, int count);
	
	public native int setKernelConfigurationParameterAsInt        (int id, int ndx, String name, int       value);
	public native int setKernelConfigurationParameterAsFloat      (int id, int ndx, String name, float     value);
	public native int setKernelConfigurationParameterAsIntArray   (int id, int ndx, String name, int    [] value);
	public native int setKernelConfigurationParameterAsFloatArray (int id, int ndx, String name, float  [] value);
	public native int setKernelConfigurationParameterAsDouble     (int id, int ndx, String name, double    value);
	
	public native int setKernelScalars (int id, int count);
	
	public native int setKernelScalarAsInt    (int id, int ndx, String name, int    value);
	public native int setKernelScalarAsFloat  (int id, int ndx, String name, float  value);
	public native int setKernelScalarAsDouble (int id, int ndx, String name, double value);
	
	/* === [cuDNN kernel configuration] === */
	
	public native int cudnnSetKernelType (int id, int type);
	
	/* 4D generic input & output descriptors */
	public native int cudnnSetKernelInputDescriptor  (int id, int count, int channels, int height, int width);
	public native int cudnnSetKernelOutputDescriptor (int id, int count, int channels, int height, int width);
	
	/* Conv */
	public native int cudnnSetConvolutionDescriptor  (int id, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth);
	
	/* Convolution model variables */
	public native int cudnnSetConvolutionFilterDescriptor (int id, int count, int channels, int height, int width);
	public native int cudnnSetConvolutionBiasDescriptor   (int id, int count, int channels, int height, int width);
	
	public native int cudnnConfigureConvolutionForwardAlgorithm (int id, int limit, double threshold);
	
	public native int cudnnConfigureConvolutionBackwardFilterAlgorithm (int id, int limit, double threshold);
	
	public native int cudnnConfigureConvolutionBackwardDataAlgorithm (int id, int limit, double threshold);
	
	/* Pool */
	public native int cudnnSetPoolingMode (int id, int mode);
	public native int cudnnSetPoolingDescriptor (int id, int windowHeight, int windowWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth);
	
	/* ReLU specialised descriptors */
	public native int cudnnSetActivationDescriptor (int id, int mode, double ceiling);
	
	/* SoftMax has no specialised descriptors */
	
	/* Batch norm */
	public native int cudnnSetBatchNormDescriptor (int id);
	public native int cudnnSetBatchNormEstimatedMeanAndVariance (int id, int capacity);
	
	/* Dropout */
	public native int cudnnSetDropoutDescriptor (int id, float dropout, long seed);
	public native int cudnnGetDropoutReserveSpaceSize (int id);
	
	/* === [End of cuDNN kernel configuration] === */
	
	public native int setDataflowGraph (int id, int [] ops);
	
	public native int setDataflowStream (int id, int op, int stream);
	
	public native int setDataflowDependency (int id, int op, int type, int guard, boolean internal);
	
	public native int setDataflowUpstreamNeighbours   (int id, int op, int [] neighbours);
	public native int setDataflowDownstreamNeighbours (int id, int op, int [] neighbours);
	
	public native int setDataflowLossOperator (int id, int op);
	public native int setDataflowAccuracyOperator (int id, int op);
	public native int setDataflowDataTransformOperator (int id, int op);
	
	public native int setDataflowPeers (int id, int [] peers);
	
	public native int setDataflowMemoryPlan (int id, int op, int provider, int position);
	
	public native int setModel (int variables, int size);
	public native int setModelVariable (int id, int order, int [] shape, int capacity);
	
	public native int setModelVariableBuffer (int id, int order, ByteBuffer buffer);
	public native int setModelVariableLearningRateMultiplier (int id, int order, float multiplier);
	
	public native int setModelWorkPerClock (int wpc);
	
	public native int setUpdateModelType (int type);
	
	/* Configure solver */
	
	public native int setLearningRateDecayPolicyFixed     (float rate);
	public native int setLearningRateDecayPolicyInv       (float rate, double gamma, double power);
	public native int setLearningRateDecayPolicyStep      (float rate, double gamma, int step);
	public native int setLearningRateDecayPolicyMultiStep (float rate, double gamma, int [] step);
	public native int setLearningRateDecayPolicyExp       (float rate, double gamma);
	public native int setLearningRateDecayPolicyCircular  (float [] rate, int superconvergence, float [] momentum, int step);
	
	public native int setBaseModelMomentum (float momentum);
	
	public native int setMomentum    (float momentum, int method);
	public native int setWeightDecay (float decay);
	
	public native int setEamsgdAlpha (float alpha);
	public native int setEamsgdTau   (int tau);
	
	public native int setModelManager (int size, int type);
	
	public native Integer acquireAccess (int [] clock);
	public native int release (Integer replica);
	public native Integer upgradeAccess (Integer replica, int [] clock);
	
	/* */
	private native int execute (
			int dataflowId,
			int taskId,
			IDataBuffer A, int startA, int endA, 
			IDataBuffer B, int startB, int endB, 
			long [] free,
			int phase, 
			Integer replicaId);
	
	public native int schedule (
			int dataflowId,
			int taskId,
			MappedDataBuffer A, long startA, long endA, 
			MappedDataBuffer B, long startB, long endB, 
			long [] free,
			int phase, 
			int bound);
	
	public native int scheduleNext (
			int dataflowId,
			int taskId,
			long startA, long endA, 
			long startB, long endB, 
			long [] free,
			int phase, 
			int bound);
	
	/* Result handler */
	public native int setResultHandler (int id, ByteBuffer slots, int count);
	
	/* Light-weight dataset handler */
	public native int setLightWeightDatasetHandler (int id, ByteBuffer slots, int count);
	
	/* Model synchronisation */
	public native int lockAny ();
	public native int merge (boolean pull);
	public native int synchronise (int first, int clock, int autotune, boolean push);
	public native int unlockAny ();
	
	public native int checkpointModel   (String directory);
	public native int overrideModelData (String directory);
	
	public native int addModel ();
	public native int delModel ();
	
	/* Record dataset(s) */
	public native int recordDatasetInit       (int phase, int workers, int [] capacity, int NB, int b, int [] padding);
	public native int recordDatasetRegister   (int phase, int id, String filename);
	public native int recordDatasetFinalise   (int phase);
}
