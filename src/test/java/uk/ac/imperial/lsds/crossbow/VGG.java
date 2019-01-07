package uk.ac.imperial.lsds.crossbow;

import java.util.HashMap;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.*;
import uk.ac.imperial.lsds.crossbow.kernel.conf.*;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.*;

public class VGG {

	private static HashMap<String, Integer> layers = new HashMap<String, Integer>();
	
	private static float EPSILON = 0.001F;
	private static float   ALPHA = 0.99F;
	
	private static String getName (String key) {
		int count = 1;
		Integer value = layers.get(key);
		if (value != null)
			count = value.intValue();
		String name = String.format("%s-%02d", key, count);
		/* Increment counter */
		layers.put(key, count + 1);
		return name;
	}
	
	private static void Convolution (DataflowNode [] f, DataflowNode [] g, int filters, int kernel, int stride) {
		
		DataflowNode node, gradient;
		
		ConvConf conf = new ConvConf ();
		
		conf.setNumberOfOutputs (filters);

		conf.setKernelSize  (2).setKernelHeight  (kernel).setKernelWidth  (kernel);
		conf.setStrideSize  (2).setStrideHeight  (stride).setStrideWidth  (stride);
		conf.setPaddingSize (2).setPaddingHeight (  1   ).setPaddingWidth (  1   );

		conf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER  ).setNorm(VarianceNormalisation.AVG)); /* Uniform within range */
		conf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		
		node = new DataflowNode (new Operator (getName("Conv"), new Conv (conf)));
		gradient = new DataflowNode (new Operator (getName("ConvGradient"), new ConvGradient (conf)));
		
		gradient.getOperator().setPeer(node.getOperator());
		
		/* Training dataflow */
		if (f[0] != null)
			f[0] = f[0].connectTo(node);
		else
			f[0] = node;
		
		if (g[0] != null)
			gradient.connectTo(g[0]);
		g[0] = gradient;
		
		/* Test dataflow */
		if (f[1] != null)
			f[1] = f[1].connectTo(node.shallowCopy());
		else
			f[1] = node.shallowCopy();
		
		return;
	}
	
	private static void BatchNormalisation (DataflowNode [] f, DataflowNode [] g) {
		
		DataflowNode node, gradient;
		
		BatchNormConf conf = new BatchNormConf ();
		
		conf.setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		
		conf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		conf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));
		
		node = new DataflowNode (new Operator (getName("BatchNorm"), new BatchNorm (conf)));
		gradient = new DataflowNode (new Operator (getName("BatchNormGradient"), new BatchNormGradient (conf)));
		
		gradient.getOperator().setPeer(node.getOperator());
		
		/* Training dataflow */
		if (f[0] != null)
			f[0] = f[0].connectTo(node);
		else
			f[0] = node;
		
		if (g[0] != null)
			gradient.connectTo(g[0]);
		g[0] = gradient;
		
		/* Test dataflow */
		if (f[1] != null)
			f[1] = f[1].connectTo(node.shallowCopy());
		else
			f[1] = node.shallowCopy();
		
		return;
	}
	
	private static void Pooling (DataflowNode [] f, DataflowNode [] g, int kernel, int stride) {
			
		DataflowNode node, gradient;
			
		PoolConf conf = new PoolConf ().setMethod (PoolMethod.MAX);
		
		conf.setKernelSize (kernel);
		conf.setStrideSize (stride);
		conf.setPaddingSize(   0  );
		
		node = new DataflowNode (new Operator (getName("Pool"), new Pool (conf)));
		gradient = new DataflowNode (new Operator (getName("PoolGradient"), new PoolGradient (conf)));
		
		gradient.getOperator().setPeer(node.getOperator());
		
		/* Training dataflow */
		if (f[0] != null)
			f[0] = f[0].connectTo(node);
		else
			f[0] = node;
		
		if (g[0] != null)
			gradient.connectTo(g[0]);
		g[0] = gradient;
		
		/* Test dataflow */
		if (f[1] != null)
			f[1] = f[1].connectTo(node.shallowCopy());
		else
			f[1] = node.shallowCopy();
		
		return;
	}
	
	private static void ReLU (DataflowNode [] f, DataflowNode [] g) {
		
		DataflowNode node, gradient;
		
		ReLUConf conf = new ReLUConf ();
		
		node = new DataflowNode (new Operator (getName("ReLU"), new ReLU (conf)));
		gradient = new DataflowNode (new Operator (getName("ReLUGradient"), new ReLUGradient (conf)));
		
		gradient.getOperator().setPeer(node.getOperator());
		
		/* Training dataflow */
		if (f[0] != null)
			f[0] = f[0].connectTo(node);
		else
			f[0] = node;
		
		if (g[0] != null)
			gradient.connectTo(g[0]);
		g[0] = gradient;
		
		/* Test dataflow */
		if (f[1] != null)
			f[1] = f[1].connectTo(node.shallowCopy());
		else
			f[1] = node.shallowCopy();
		
		return;
	}
	
	private static void Dropout (DataflowNode [] f, DataflowNode [] g, float ratio) {
		
		DataflowNode node, gradient;
		
		DropoutConf conf = new DropoutConf ().setRatio(ratio);
		
		node = new DataflowNode (new Operator (getName("Dropout"), new Dropout (conf)));
		gradient = new DataflowNode (new Operator (getName("DropoutGradient"), new DropoutGradient (conf)));
		
		gradient.getOperator().setPeer(node.getOperator());
		
		/* Training dataflow */
		if (f[0] != null)
			f[0] = f[0].connectTo(node);
		else
			f[0] = node;
		
		if (g[0] != null)
			gradient.connectTo(g[0]);
		g[0] = gradient;
		
		/* Test dataflow */
		if (f[1] != null)
			f[1] = f[1].connectTo(node.shallowCopy());
		else
			f[1] = node.shallowCopy();
		
		return;
	}
	
	private static void InnerProduct (DataflowNode [] f, DataflowNode [] g, int outputs) {
		
		DataflowNode node, gradient;
		
		InnerProductConf conf = new InnerProductConf ();
		
		conf.setNumberOfOutputs   (outputs);
		conf.setBias              (true);
		conf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER  ).setNorm(VarianceNormalisation.AVG)); /* Uniform within range */
		conf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		
		node = new DataflowNode (new Operator (getName("InnerProduct"), new InnerProduct (conf)));
		gradient = new DataflowNode (new Operator (getName("InnerProductGradient"), new InnerProductGradient (conf)));
		
		gradient.getOperator().setPeer(node.getOperator());
		
		/* Training dataflow */
		if (f[0] != null)
			f[0] = f[0].connectTo(node);
		else
			f[0] = node;
		
		if (g[0] != null)
			gradient.connectTo(g[0]);
		g[0] = gradient;
		
		/* Test dataflow */
		if (f[1] != null)
			f[1] = f[1].connectTo(node.shallowCopy());
		else
			f[1] = node.shallowCopy();
		
		return;
	}
	
	public static SubGraph [] buildVGG (SolverConf solverConf, int outputs) {
		
		SubGraph [] graphs = new SubGraph [] { null, null }; /* Return value */
		
		DataflowNode [] f = new DataflowNode [] { null, null };
		DataflowNode [] g = new DataflowNode [] { null, null };
		
		DataflowNode [] h = new DataflowNode [] { null, null };
		DataflowNode [] t = new DataflowNode [] { null, null };
		
		/* Stage 0: Data transformations */
		DataTransformConf datatransformConf = new DataTransformConf ();
		datatransformConf.setCropSize (32);
		datatransformConf.setMirror (true);
		
		DataflowNode datatransform = 
			new DataflowNode (new Operator ("DataTransform", new DataTransform (datatransformConf)));
		
		/* Since this is a data transformation, set head only */
		f[0] = datatransform;
		f[1] = datatransform.shallowCopy ();
		h[0] = f[0];
		h[1] = f[1];
		
		/* Stage 1 */
		Convolution (f, g, 64, 3, 1);
		/* Since this is the first operator with a gradient, set the tail */
		t[0] = g[0];
		t[1] = g[1];
		/* Continue building the graphs(s) */
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.3F);
		Convolution (f, g, 64, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Pooling (f, g, 2, 2);
		
		/* Stage 2 */
		Convolution (f, g, 128, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 128, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Pooling (f, g, 2, 2);
		
		/* Stage 3 */
		Convolution (f, g, 256, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 256, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 256, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Pooling (f, g, 2, 2);
		
		/* Stage 4 */
		Convolution (f, g, 512, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 512, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 512, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Pooling (f, g, 2, 2);
		
		/* Stage 5 */
		Convolution (f, g, 512, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 512, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.4F);
		Convolution (f, g, 512, 3, 1);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Pooling (f, g, 2, 2);
		
		/* Final stage */
		Dropout (f, g, 0.5F);
		InnerProduct (f, g, 512);
		ReLU (f, g);
		BatchNormalisation (f, g);
		Dropout (f, g, 0.5F);
		InnerProduct (f, g, outputs);
		
		/* Last few nodes */
		
		SoftMaxConf softmaxConf = new SoftMaxConf ();
		LossConf lossConf = new LossConf ();
		AccuracyConf accuracyConf = new AccuracyConf ();
		
		DataflowNode softmax = new DataflowNode (new Operator ("SoftMax", new SoftMax (softmaxConf)));
		DataflowNode loss = new DataflowNode (new Operator ("SoftMaxLoss", new SoftMaxLoss (lossConf)));
		DataflowNode accuracy = new DataflowNode (new Operator ("Accuracy", new Accuracy (accuracyConf)));
		
		DataflowNode lossGradient = new DataflowNode (new Operator ("SoftMaxLossGradient", new SoftMaxLossGradient (lossConf)));
		
		lossGradient.getOperator().setPeer(loss.getOperator());
		
		/* Create solver operator */
		DataflowNode optimiser = new DataflowNode (new Operator ("Optimiser", new GradientDescentOptimiser (solverConf)));
		
		/* Finish training dataflow */
		
		f[0] = f[0].connectTo(softmax).connectTo(loss);
		softmax.connectTo(accuracy);
		
		lossGradient.connectTo(g[0]);
		g[0] = lossGradient;
		
		/* Connect f (the tail of nodes) with g (the head of gradients) */
		f[0].connectTo(g[0]);
		
		/* Connect the tail of g with the optimiser */
		t[0].connectTo(optimiser);
		
		/* Finish test dataflow */
		
		DataflowNode softmax_ = softmax.shallowCopy();
		f[1] = f[1].connectTo(softmax_).connectTo(loss.shallowCopy());
		softmax_.connectTo(accuracy.shallowCopy());
		
		/* Done */
		
		graphs [0] = new SubGraph (h [0]);
		graphs [1] = new SubGraph (h [1]);
		
		return graphs;
	}
	
	public static void main (String [] args) throws Exception {
		
		long startTime, dt;
		
		Options options = new Options (VGG.class.getName());
		
		options.addOption ("--training-unit",  "Training unit",     true,  String.class, "epochs");
		options.addOption ("--N",              "Train for N units", true, Integer.class,    "250");
		options.addOption ("--data-directory", "Data directory",    true,  String.class,       "");
		
		CommandLine commandLine = new CommandLine (options);
		
		int numberofreplicas = 1;
		int [] devices = new int [] { 0 };
		int wpc = numberofreplicas * devices.length;
		
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (numberofreplicas)
			.setNumberOfGPUStreams (numberofreplicas)
			.setNumberOfGPUCallbackHandlers (8)
			.setNumberOfGPUTaskHandlers (1)
			.setMaxNumberOfMappedPartitions (1)
			.setDisplayInterval (100)
			.setDisplayIntervalUnit (TrainingUnit.TASKS)
			.displayAccumulatedLossValue (true)
			.setRandomSeed (123456789L)
			.queueMeasurements (true)
			.setTaskQueueSizeLimit (32)
			.setPerformanceMonitorInterval (1000)
			.setNumberOfResultSlots (128)
			.setGPUDevices (devices)
			.allowMemoryReuse(true)
			.setNumberOfFileHandlers(4)
			.autotuneModels(false);
			
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (128).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.WORKER);
		ModelConf.getInstance ().getSolverConf ().setAlpha (0.1f).setTau (3);
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.1F)
			.setMomentum (0.9F)
			.setMomentumMethod(MomentumMethod.POLYAK)
			.setWeightDecay(0.0005F)
			.setLearningRateStepUnit(TrainingUnit.EPOCHS)
			.setStepValues(new int [] { 82, 122 })
			.setGamma(0.1F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int __N = options.getOption("--N").getIntValue ();
		TrainingUnit taskunit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		SolverConf solverConf = ModelConf.getInstance().getSolverConf ();
		
		/* Build dataflows for both training and testing phases */
		SubGraph [] graphs = null;
		
		IDataset []  dataset = new IDataset [] { null, null };
		
		/* Set dataflow */

		Dataflow [] dataflows = new Dataflow [2];
			
		/* Create dataset */
		String dataDirectory = options.getOption("--data-directory").getStringValue();
		
		dataset [0] = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));
		dataset [1] = new Dataset (DatasetUtils.buildPath(dataDirectory,  "cifar-test.metadata", true));
		
		graphs = buildVGG (solverConf, 100);
			
		dataflows [0] = new Dataflow (graphs [0]).setPhase(Phase.TRAIN);
		dataflows [1] = new Dataflow (graphs [1]).setPhase(Phase.CHECK);
		
		/* Model dataset configuration */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, dataset[0]).setDataset (Phase.CHECK, dataset[1]);
		
		if (! (wpc > 0))
			throw new IllegalArgumentException();
		
		ExecutionContext context = new ExecutionContext (dataflows);

		context.init();
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		
		startTime = System.nanoTime ();
		
		try {
			if (__N > 0) {
				if (dataset [1] == null) {
					context.train(__N, taskunit);
				} 
				else {
					context.trainAndTest(__N, taskunit);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
		
		if (SystemConf.getInstance().queueMeasurements()) {
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();
			
			if (dataset [1] != null) {
				context.getDataflow(Phase.CHECK).getResultHandler().getMeasurementQueue().dump();
			}
		}
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
