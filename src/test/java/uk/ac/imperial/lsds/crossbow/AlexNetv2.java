package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.*;
import uk.ac.imperial.lsds.crossbow.kernel.conf.*;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.*;

public class AlexNetv2 {

	/* Reference : TF */

	public static final String usage = "usage: AlexNetV2";

	private final static Logger log = LogManager.getLogger (AlexNetv2.class);

	public static void main (String [] args) throws Exception {

		long startTime, dt;

		Options options = new Options (AlexNetv2.class.getName());

		options.addOption ("--training-unit",   "Training unit",     true,   	String.class,  		"epochs");
		options.addOption ("--N",             	"Train for N units", true,  	Integer.class,   	"164");
		options.addOption ("--target-loss",   	"Target loss",       false,   	Float.class,       	"0");

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
				.setDisplayInterval (1)
				.setDisplayIntervalUnit (TrainingUnit.EPOCHS)
				.displayAccumulatedLossValue (true)
				.setRandomSeed(123456789L)
				.queueMeasurements(true)
				.setTaskQueueSizeLimit(16)
				.setPerformanceMonitorInterval(1000)
				.setNumberOfResultSlots(128)
				.setGPUDevices(devices)
				.allowMemoryReuse(true);

		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (128).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.DEFAULT);
		ModelConf.getInstance ().getSolverConf ().setAlpha (0.5F).setTau (3);
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);

		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
				.setLearningRateDecayPolicy (LearningRateDecayPolicy.MULTISTEP)
				.setBaseLearningRate (0.1F)
				.setMomentum (0.9F)
				.setWeightDecay(0.0001F)
				.setLearningRateStepUnit(TrainingUnit.EPOCHS)
				.setStepValues(new int [] { 82, 122})
				.setGamma(0.1F);

		/* Parse command line arguments */
		commandLine.parse (args);

		int __N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());

		/* Parse command line arguments */
		commandLine.parse (args);

		int batch_size     = ModelConf.getInstance().getBatchSize();

		SolverConf solverConf = ModelConf.getInstance().getSolverConf ();

		int cropsize = 224;
		int numclasses = 10;
		String meanimage = null;
		
		DataTransformConf transformConf = new DataTransformConf ();
		transformConf.setCropSize (cropsize);
		transformConf.setMeanImageFilename (meanimage);
		
		ConvConf convConf_s1 = new ConvConf ().setNumberOfOutputs (96);
		
		convConf_s1.setKernelSize (2).setKernelHeight (11).setKernelWidth (11);
		convConf_s1.setStrideSize (2).setStrideHeight  (4).setStrideWidth  (4);
		
		convConf_s1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.01F));
		convConf_s1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		
		PoolConf poolConf_s1 = new PoolConf ();
		
		poolConf_s1.setKernelSize  (3);
		poolConf_s1.setStrideSize  (2);
		poolConf_s1.setPaddingSize (0);
		
		ReLUConf reluConf_s1 = new ReLUConf ();
		
		ConvConf convConf_s2 = new ConvConf ().setNumberOfOutputs (256);
		
		convConf_s2.setKernelSize  (2).setKernelHeight  (5).setKernelWidth  (5);
		convConf_s2.setPaddingSize (2).setPaddingHeight (2).setPaddingWidth (2);
		
		convConf_s2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.01F));
		convConf_s2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.1F));
		
		PoolConf poolConf_s2 = new PoolConf ();
		
		poolConf_s2.setKernelSize  (3);
		poolConf_s2.setStrideSize  (2);
		poolConf_s2.setPaddingSize (0);
		
		ReLUConf reluConf_s2 = new ReLUConf ();
		
		ConvConf convConf_s3 = new ConvConf ().setNumberOfOutputs (384);
		
		convConf_s3.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convConf_s3.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convConf_s3.setWeightInitialiser (new InitialiserConf ().setType(InitialiserType.GAUSSIAN).setStd(0.01F));
		convConf_s3.setBiasInitialiser   (new InitialiserConf ().setType(InitialiserType.CONSTANT).setValue(0));
		
		ReLUConf reluConf_s3 = new ReLUConf ();
		
		ConvConf convConf_s4 = new ConvConf ().setNumberOfOutputs (384);
		
		convConf_s4.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convConf_s4.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convConf_s4.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.010F));
		convConf_s4.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.1F));
		
		ReLUConf reluConf_s4 = new ReLUConf ();
		
		ConvConf convConf_s5 = new ConvConf ().setNumberOfOutputs (256);
		
		convConf_s5.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convConf_s5.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convConf_s5.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.010F));
		convConf_s5.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.1F));
		
		PoolConf poolConf_s5 = new PoolConf ();
		
		poolConf_s5.setKernelSize  (3);
		poolConf_s5.setStrideSize  (2);
		poolConf_s5.setPaddingSize (0);
		
		ReLUConf reluConf_s5 = new ReLUConf ();
		
		InnerProductConf ipConf_s6 = new InnerProductConf ().setNumberOfOutputs (4096);
		
		ipConf_s6.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.005F));
		ipConf_s6.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.1F));
		
		ReLUConf reluConf_s6 = new ReLUConf();
		
		InnerProductConf ipConf_s7 = new InnerProductConf().setNumberOfOutputs(4096);
		
		ipConf_s7.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.005F));
		ipConf_s7.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.1F));
		
		ReLUConf reluConf_s7 = new ReLUConf ();
		
		InnerProductConf ipConf = new InnerProductConf ().setNumberOfOutputs (numclasses);
		
		ipConf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.010F));
		ipConf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.0F));
		
		SoftMaxConf softMaxConf = new SoftMaxConf ();
		LossConf lossConf = new LossConf ();
		

		
		AccuracyConf accuracyConf = new AccuracyConf ();
		
//		ModelConf.getInstance().setSolverConf (solverConf);
		
		Operator   dataTransform = new Operator (                 "DataTransform", new          DataTransform (transformConf));
		
		Operator         conv_s1 = new Operator (                "Conv-stage-1", new                     Conv (  convConf_s1));
		Operator         relu_s1 = new Operator (                "ReLU-stage-1", new                     ReLU (  reluConf_s1));
		Operator         pool_s1 = new Operator (                "Pool-stage-1", new                     Pool (  poolConf_s1));
		
		Operator         conv_s2 = new Operator (                "Conv-stage-2", new                     Conv (  convConf_s2));
		Operator         relu_s2 = new Operator (                "ReLU-stage-2", new                     ReLU (  reluConf_s2));
		Operator         pool_s2 = new Operator (                "Pool-stage-2", new                     Pool (  poolConf_s2));
		
		Operator         conv_s3 = new Operator (                "Conv-stage-3", new                     Conv (  convConf_s3));
		Operator         relu_s3 = new Operator (                "ReLU-stage-3", new                     ReLU (  reluConf_s3));
		
		Operator         conv_s4 = new Operator (                "Conv-stage-4", new                     Conv (  convConf_s4));
		Operator         relu_s4 = new Operator (                "ReLU-stage-4", new                     ReLU (  reluConf_s4));
		
		Operator         conv_s5 = new Operator (                "Conv-stage-5", new                     Conv (  convConf_s5));
		Operator         relu_s5 = new Operator (                "ReLU-stage-5", new                     ReLU (  reluConf_s5));
		Operator         pool_s5 = new Operator (                "Pool-stage-5", new                     Pool (  poolConf_s5));
		
		Operator           ip_s6 = new Operator (        "InnerProduct-stage-6", new             InnerProduct (    ipConf_s6));
		Operator         relu_s6 = new Operator (                "ReLU-stage-6", new                     ReLU (  reluConf_s6));
		
		Operator           ip_s7 = new Operator (        "InnerProduct-stage-7", new             InnerProduct (    ipConf_s7));
		Operator         relu_s7 = new Operator (                "ReLU-stage-7", new                     ReLU (  reluConf_s7));
		
		Operator              ip = new Operator (                "InnerProduct", new             InnerProduct (       ipConf));
		Operator         softmax = new Operator (                     "SoftMax", new                  SoftMax (  softMaxConf));
		Operator            loss = new Operator (                 "SoftMaxLoss", new              SoftMaxLoss (     lossConf));
		
		Operator    lossGradient = new Operator (         "SoftMaxLossGradient", new      SoftMaxLossGradient (     lossConf)).setPeer(   loss);
		Operator      ipGradient = new Operator (        "InnerProductGradient", new     InnerProductGradient (       ipConf)).setPeer(     ip);
		
		Operator reluGradient_s7 = new Operator (        "ReLUGradient-stage-7", new             ReLUGradient (  reluConf_s7)).setPeer(relu_s7);
		Operator   ipGradient_s7 = new Operator ("InnerProductGradient-stage-7", new     InnerProductGradient (    ipConf_s7)).setPeer(  ip_s7);
		
		Operator reluGradient_s6 = new Operator (        "ReLUGradient-stage-6", new             ReLUGradient (  reluConf_s6)).setPeer(relu_s6);
		Operator   ipGradient_s6 = new Operator ("InnerProductGradient-stage-6", new     InnerProductGradient (    ipConf_s6)).setPeer(  ip_s6);
		
		Operator poolGradient_s5 = new Operator (        "PoolGradient-stage-5", new             PoolGradient (  poolConf_s5)).setPeer(pool_s5);
		Operator reluGradient_s5 = new Operator (        "ReLUGradient-stage-5", new             ReLUGradient (  reluConf_s5)).setPeer(relu_s5);
		Operator convGradient_s5 = new Operator (        "ConvGradient-stage-5", new             ConvGradient (  convConf_s5)).setPeer(conv_s5);
		
		Operator reluGradient_s4 = new Operator (        "ReLUGradient-stage-4", new             ReLUGradient (  reluConf_s4)).setPeer(relu_s4);
		Operator convGradient_s4 = new Operator (        "ConvGradient-stage-4", new             ConvGradient (  convConf_s4)).setPeer(conv_s4);
		
		Operator reluGradient_s3 = new Operator (        "ReLUGradient-stage-3", new             ReLUGradient (  reluConf_s3)).setPeer(relu_s3);
		Operator convGradient_s3 = new Operator (        "ConvGradient-stage-3", new             ConvGradient (  convConf_s3)).setPeer(conv_s3);
		
		Operator poolGradient_s2 = new Operator (        "PoolGradient-stage-2", new             PoolGradient (  poolConf_s2)).setPeer(pool_s2);
		Operator reluGradient_s2 = new Operator (        "ReLUGradient-stage-2", new             ReLUGradient (  reluConf_s2)).setPeer(relu_s2);
		Operator convGradient_s2 = new Operator (        "ConvGradient-stage-2", new             ConvGradient (  convConf_s2)).setPeer(conv_s2);
		
		Operator poolGradient_s1 = new Operator (        "PoolGradient-stage-1", new             PoolGradient (  poolConf_s1)).setPeer(pool_s1);
		Operator reluGradient_s1 = new Operator (        "ReLUGradient-stage-1", new             ReLUGradient (  reluConf_s1)).setPeer(relu_s1);
		Operator convGradient_s1 = new Operator (        "ConvGradient-stage-1", new             ConvGradient (  convConf_s1)).setPeer(conv_s1);
		
		Operator       optimiser = new Operator (                   "Optimiser", new GradientDescentOptimiser (   solverConf));
		Operator        accuracy = new Operator (                    "Accuracy", new                 Accuracy ( accuracyConf));
		
		DataflowNode h1 = new DataflowNode (dataTransform);
		h1
			.connectTo(new DataflowNode (        conv_s1))
			.connectTo(new DataflowNode (        relu_s1))
			.connectTo(new DataflowNode (        pool_s1))
			.connectTo(new DataflowNode (        conv_s2))
			.connectTo(new DataflowNode (        relu_s2))
			.connectTo(new DataflowNode (        pool_s2))
			.connectTo(new DataflowNode (        conv_s3))
			.connectTo(new DataflowNode (        relu_s3))
			.connectTo(new DataflowNode (        conv_s4))
			.connectTo(new DataflowNode (        relu_s4))
			.connectTo(new DataflowNode (        conv_s5))
			.connectTo(new DataflowNode (        relu_s5))
			.connectTo(new DataflowNode (        pool_s5))
			.connectTo(new DataflowNode (          ip_s6))
			.connectTo(new DataflowNode (        relu_s6))
			.connectTo(new DataflowNode (          ip_s7))
			.connectTo(new DataflowNode (        relu_s7))
			.connectTo(new DataflowNode (             ip))
			.connectTo(new DataflowNode (        softmax))
			.connectTo(new DataflowNode (           loss))
			.connectTo(new DataflowNode (   lossGradient))
			.connectTo(new DataflowNode (     ipGradient))
			.connectTo(new DataflowNode (reluGradient_s7))
			.connectTo(new DataflowNode (  ipGradient_s7))
			.connectTo(new DataflowNode (reluGradient_s6))
			.connectTo(new DataflowNode (  ipGradient_s6))
			.connectTo(new DataflowNode (poolGradient_s5))
			.connectTo(new DataflowNode (reluGradient_s5))
			.connectTo(new DataflowNode (convGradient_s5))
			.connectTo(new DataflowNode (reluGradient_s4))
			.connectTo(new DataflowNode (convGradient_s4))
			.connectTo(new DataflowNode (reluGradient_s3))
			.connectTo(new DataflowNode (convGradient_s3))
			.connectTo(new DataflowNode (poolGradient_s2))
			.connectTo(new DataflowNode (reluGradient_s2))
			.connectTo(new DataflowNode (convGradient_s2))
			.connectTo(new DataflowNode (poolGradient_s1))
			.connectTo(new DataflowNode (reluGradient_s1))
			.connectTo(new DataflowNode (convGradient_s1))
			.connectTo(new DataflowNode (      optimiser));

		DataflowNode h2 = new DataflowNode (dataTransform);
		DataflowNode node = h2
			.connectTo(new DataflowNode (        conv_s1))
			.connectTo(new DataflowNode (        relu_s1))
			.connectTo(new DataflowNode (        pool_s1))
			.connectTo(new DataflowNode (        conv_s2))
			.connectTo(new DataflowNode (        relu_s2))
			.connectTo(new DataflowNode (        pool_s2))
			.connectTo(new DataflowNode (        conv_s3))
			.connectTo(new DataflowNode (        relu_s3))
			.connectTo(new DataflowNode (        conv_s4))
			.connectTo(new DataflowNode (        relu_s4))
			.connectTo(new DataflowNode (        conv_s5))
			.connectTo(new DataflowNode (        relu_s5))
			.connectTo(new DataflowNode (        pool_s5))
			.connectTo(new DataflowNode (          ip_s6))
			.connectTo(new DataflowNode (        relu_s6))
			.connectTo(new DataflowNode (          ip_s7))
			.connectTo(new DataflowNode (        relu_s7))
			.connectTo(new DataflowNode (             ip))
			.connectTo(new DataflowNode (        softmax));
		
		node.connectTo(new DataflowNode (           loss));
		node.connectTo(new DataflowNode (       accuracy));
		
		SubGraph g1 = new SubGraph (h1);
		SubGraph g2 = new SubGraph (h2);

		Dataflow [] dataflows = new Dataflow [2];

		dataflows [0] = new Dataflow (g1).setPhase(Phase.TRAIN);
		dataflows [1] = new Dataflow (g2).setPhase(Phase.CHECK);

		Dataset []  dataset = new Dataset [] { null, null };

		String dataDirectory = String.format("/data/crossbow/imagenet10/b-%03d/", batch_size);
		dataset [0] = new Dataset (DatasetUtils.buildPath(dataDirectory, "imagenet-train.metadata", true));
		dataset [1] = new Dataset (DatasetUtils.buildPath(dataDirectory, "imagenet-train.metadata", true));

		ModelConf.getInstance ().setBatchSize (batch_size).setDataset (Phase
				.TRAIN, dataset[0]).setDataset (Phase.CHECK, dataset[1]);

		int [] tasksize = ModelConf.getInstance ().getTaskSize ();

		log.info(String.format("%d examples/task; %d tasks/epoch; %d and %d bytes/task for examples and labels respectively",
				ModelConf.getInstance().getBatchSize(),
				ModelConf.getInstance().numberOfTasksPerEpoch(),
				tasksize[0],
				tasksize[1]
		));

		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();

		ExecutionContext context = new ExecutionContext (dataflows);

		context.init();

		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();

		startTime = System.nanoTime ();

		if (__N > 0) {
			try {

				context.train(__N, unit);

			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));

		if (SystemConf.getInstance().queueMeasurements())
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();

		context.destroy();

		System.out.println("Bye.");
		System.exit(0);
	}
}
