package uk.ac.imperial.lsds.crossbow;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.Accuracy;
import uk.ac.imperial.lsds.crossbow.kernel.Conv;
import uk.ac.imperial.lsds.crossbow.kernel.ConvGradient;
import uk.ac.imperial.lsds.crossbow.kernel.GradientDescentOptimiser;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProduct;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProductGradient;
import uk.ac.imperial.lsds.crossbow.kernel.Pool;
import uk.ac.imperial.lsds.crossbow.kernel.PoolGradient;
import uk.ac.imperial.lsds.crossbow.kernel.ReLU;
import uk.ac.imperial.lsds.crossbow.kernel.ReLUGradient;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMax;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLoss;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLossGradient;
import uk.ac.imperial.lsds.crossbow.kernel.conf.AccuracyConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConvConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ReLUConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.DurationUnit;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

public class ConvNet {
	
	public static final String usage = "usage: ConvNet";
	
	// private final static Logger log = LogManager.getLogger (ConvNet.class);
	
	public static void main (String [] args) throws Exception {
        
        long startTime, dt;
		
		Options options = new Options (LeNet.class.getName());
		
		options.addOption ("--target-loss",   	"Target loss",            false,   Float.class,        "0");
		options.addOption ("--time-limit",   	"Time-limited training",  true,  Boolean.class,    "false");
		options.addOption ("--training-unit",   "Training unit",          true,   String.class,   "epochs");
		options.addOption ("--N",             	"Train for N units",      true,  Integer.class,      "200");
		options.addOption ("--duration-unit",   "Duration time unit",     true,   String.class,  "minutes");
		options.addOption ("--D",   	        "Train for D time units", true,  Integer.class,        "1");
		
		/* Switch between xavier and constant weight initialisation */
		options.addOption ("--weight-initialiser", "Weight initialisation type", false, String.class, "gaussian");
		
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
			.setDisplayInterval (937)
			.displayAccumulatedLossValue (true)
			.queueMeasurements (true)
			.setGPUDevices(devices);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (128).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.DEFAULT);
		ModelConf.getInstance ().getSolverConf ().setAlpha (1F / wpc).setTau (3);
		
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.001F)
			.setMomentum (0.001F)
			.setWeightDecay(0.004F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit taskunit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		boolean timelimit = options.getOption("--time-limit").getBooleanValue();
		
		int D = options.getOption("--D").getIntValue ();
		DurationUnit timeunit = DurationUnit.fromString (options.getOption("--duration-unit").getStringValue());
		
		String dir = String.format("/data/crossbow/cifar-10/b-%03d/", ModelConf.getInstance().getBatchSize());
		
		Dataset ds1 = new Dataset (dir + "cifar-train.metadata");
		Dataset ds2 = new Dataset (dir + "cifar-test.metadata");
		
		/* Load dataset(s) */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, ds1).setDataset (Phase.CHECK, ds2);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
	
		
		/* Operator configuration */
		/*
		InitialiserType initialiserType = InitialiserType.fromString (options.getOption("--weight-initialiser").getStringValue());
		*/
		
		/*
		DataTransformConf  dataTransformConf = new DataTransformConf().setMeanImageFilename("/data/crossbow/cifar-10/mean.image");
		*/
		ConvConf                 _1_convConf = new          ConvConf().setNumberOfOutputs(32).setStrideSize(2).setStrideHeight(1).setStrideWidth(1).setKernelSize(2).setKernelHeight(5).setKernelWidth(5).setPaddingSize(2).setPaddingHeight(2).setPaddingWidth(2).setWeightInitialiser(new InitialiserConf().setType(InitialiserType.GAUSSIAN).setStd(0.0001F)).setBiasInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT)).setBiasLearningRateMultiplier(2);
		ConvConf                 _2_convConf = new          ConvConf().setNumberOfOutputs(32).setStrideSize(2).setStrideHeight(1).setStrideWidth(1).setKernelSize(2).setKernelHeight(5).setKernelWidth(5).setPaddingSize(2).setPaddingHeight(2).setPaddingWidth(2).setWeightInitialiser(new InitialiserConf().setType(InitialiserType.GAUSSIAN).setStd(0.01F)).setBiasInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT)).setBiasLearningRateMultiplier(2);
		ConvConf                 _3_convConf = new          ConvConf().setNumberOfOutputs(64).setStrideSize(2).setStrideHeight(1).setStrideWidth(1).setKernelSize(2).setKernelHeight(5).setKernelWidth(5).setPaddingSize(2).setPaddingHeight(2).setPaddingWidth(2).setWeightInitialiser(new InitialiserConf().setType(InitialiserType.GAUSSIAN).setStd(0.01F)).setBiasInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT)).setBiasLearningRateMultiplier(2);
		PoolConf                 _1_poolConf = new          PoolConf().setKernelSize(3).setStrideSize(2).setStrideHeight(2).setStrideWidth(2).setPaddingSize(2).setPaddingHeight(0).setPaddingWidth(0).setMethod(PoolMethod.AVERAGE);
		PoolConf                 _2_poolConf = new          PoolConf().setKernelSize(3).setStrideSize(2).setStrideHeight(2).setStrideWidth(2).setPaddingSize(2).setPaddingHeight(0).setPaddingWidth(0).setMethod(PoolMethod.AVERAGE);
		PoolConf                 _3_poolConf = new          PoolConf().setKernelSize(3).setStrideSize(2).setStrideHeight(2).setStrideWidth(2).setPaddingSize(2).setPaddingHeight(0).setPaddingWidth(0).setMethod(PoolMethod.AVERAGE);
		InnerProductConf _1_innerProductConf = new  InnerProductConf().setNumberOfOutputs(64).setWeightInitialiser(new InitialiserConf().setType(InitialiserType.GAUSSIAN).setStd(0.1F)).setBiasInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT)).setBiasLearningRateMultiplier(2);
		InnerProductConf _2_innerProductConf = new  InnerProductConf().setNumberOfOutputs(10).setWeightInitialiser(new InitialiserConf().setType(InitialiserType.GAUSSIAN).setStd(0.1F)).setBiasInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT)).setBiasLearningRateMultiplier(2);
		ReLUConf                    reluConf = new          ReLUConf();
		SoftMaxConf              softMaxConf = new       SoftMaxConf();
		LossConf                    lossConf = new          LossConf();
		AccuracyConf            accuracyConf = new      AccuracyConf();
		SolverConf                solverConf = new        SolverConf();
		
		/*
		DataflowNode     transformer = new DataflowNode(new Operator (       "DataTransform", new            DataTransform (  dataTransformConf)));
		*/
		
		DataflowNode           conv1 = new DataflowNode(new Operator (                "Conv", new                     Conv (        _1_convConf)));
		DataflowNode           pool1 = new DataflowNode(new Operator (                "Pool", new                     Pool (        _1_poolConf)));
		DataflowNode           relu1 = new DataflowNode(new Operator (                "ReLU", new                     ReLU (           reluConf)));
		
		DataflowNode           conv2 = new DataflowNode(new Operator (                "Conv", new                     Conv (        _2_convConf)));
		DataflowNode           relu2 = new DataflowNode(new Operator (                "ReLU", new                     ReLU (           reluConf)));
		DataflowNode           pool2 = new DataflowNode(new Operator (                "Pool", new                     Pool (        _2_poolConf)));
		
		DataflowNode           conv3 = new DataflowNode(new Operator (                "Conv", new                     Conv (        _3_convConf)));
		DataflowNode           relu3 = new DataflowNode(new Operator (                "ReLU", new                     ReLU (           reluConf)));
		DataflowNode           pool3 = new DataflowNode(new Operator (                "Pool", new                     Pool (        _3_poolConf)));
		
		DataflowNode             ip1 = new DataflowNode(new Operator (        "InnerProduct", new             InnerProduct (_1_innerProductConf)));
		DataflowNode             ip2 = new DataflowNode(new Operator (        "InnerProduct", new             InnerProduct (_2_innerProductConf)));
		DataflowNode         softmax = new DataflowNode(new Operator (             "SoftMax", new                  SoftMax (        softMaxConf)));
		DataflowNode            loss = new DataflowNode(new Operator (                "Loss", new      		  SoftMaxLoss (           lossConf)));
		
		DataflowNode    lossGradient = new DataflowNode(new Operator (        "LossGradient", new      SoftMaxLossGradient (           lossConf)).setPeer (   loss.getOperator()));
		DataflowNode     ipGradient2 = new DataflowNode(new Operator ("InnerProductGradient", new     InnerProductGradient (_2_innerProductConf)).setPeer (    ip2.getOperator()));
		DataflowNode     ipGradient1 = new DataflowNode(new Operator ("InnerProductGradient", new     InnerProductGradient (_1_innerProductConf)).setPeer (    ip1.getOperator()));
		
		DataflowNode   poolGradient3 = new DataflowNode(new Operator (        "PoolGradient", new             PoolGradient (        _3_poolConf)).setPeer (    pool3.getOperator()));
		DataflowNode   reluGradient3 = new DataflowNode(new Operator (        "ReLUGradient", new             ReLUGradient (           reluConf)).setPeer (    relu3.getOperator()));
		DataflowNode   convGradient3 = new DataflowNode(new Operator (        "ConvGradient", new             ConvGradient (        _3_convConf)).setPeer (    conv3.getOperator()));
		
		DataflowNode   poolGradient2 = new DataflowNode(new Operator (        "PoolGradient", new             PoolGradient (        _2_poolConf)).setPeer (    pool2.getOperator()));
		DataflowNode   reluGradient2 = new DataflowNode(new Operator (        "ReLUGradient", new             ReLUGradient (           reluConf)).setPeer (    relu2.getOperator()));
		DataflowNode   convGradient2 = new DataflowNode(new Operator (        "ConvGradient", new             ConvGradient (        _2_convConf)).setPeer (    conv2.getOperator()));
		
		DataflowNode   reluGradient1 = new DataflowNode(new Operator (        "ReLUGradient", new             ReLUGradient (           reluConf)).setPeer (    relu1.getOperator()));
		DataflowNode   poolGradient1 = new DataflowNode(new Operator (        "PoolGradient", new             PoolGradient (        _1_poolConf)).setPeer (    pool1.getOperator()));
		DataflowNode   convGradient1 = new DataflowNode(new Operator (        "ConvGradient", new             ConvGradient (        _1_convConf)).setPeer (    conv1.getOperator()));
		
		DataflowNode       optimiser = new DataflowNode(new Operator (           "Optimiser", new GradientDescentOptimiser (         solverConf)));
		
		DataflowNode        accuracy = new DataflowNode(new Operator (            "Accuracy", new                 Accuracy (       accuracyConf)));
		
		conv1.connectTo(pool1).connectTo(relu1).connectTo(conv2).connectTo(relu2).connectTo(pool2).connectTo(conv3).connectTo(relu3).connectTo(pool3).connectTo(ip1).connectTo(ip2).connectTo(softmax);
		softmax.connectTo(loss);
		softmax.connectTo(accuracy);
		loss.connectTo(lossGradient).connectTo(ipGradient2).connectTo(ipGradient1).connectTo(poolGradient3).connectTo(reluGradient3).connectTo(convGradient3).connectTo(poolGradient2).connectTo(reluGradient2)
		.connectTo(convGradient2).connectTo(reluGradient1).connectTo(poolGradient1).connectTo(convGradient1).connectTo(optimiser);
		
		
		
		/* Testing */
		
		DataflowNode           _conv1 = conv1.shallowCopy();
		DataflowNode           _pool1 = pool1.shallowCopy();
		DataflowNode           _relu1 = relu1.shallowCopy();
		DataflowNode           _conv2 = conv2.shallowCopy();
		DataflowNode           _pool2 = pool2.shallowCopy();
		DataflowNode           _relu2 = relu2.shallowCopy();
		DataflowNode           _conv3 = conv3.shallowCopy();
		DataflowNode           _pool3 = pool3.shallowCopy();
		DataflowNode           _relu3 = relu3.shallowCopy();
		DataflowNode             _ip1 = ip1.shallowCopy();
		DataflowNode             _ip2 = ip2.shallowCopy();
		DataflowNode         _softmax = softmax.shallowCopy();
		DataflowNode        _accuracy = accuracy.shallowCopy();
		
		_conv1.connectTo(_pool1).connectTo(_relu1).connectTo(_conv2).connectTo(_pool2).connectTo(_relu2).connectTo(_conv3).connectTo(_pool3).connectTo(_relu3).connectTo(_ip1).connectTo(_ip2).connectTo(_softmax).connectTo(_accuracy);
		
		SubGraph g1 = new SubGraph ( conv1);
		SubGraph g2 = new SubGraph (_conv1);
		
		Dataflow df1 = new Dataflow (g1).setPhase(Phase.TRAIN);
		Dataflow df2 = new Dataflow (g2).setPhase(Phase.CHECK);
		
		ExecutionContext context = new ExecutionContext (new Dataflow [] { df1, df2 });
		

		context.init();
		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		context.getModel().dump();
		
		
		startTime = System.nanoTime();
		
		try {
			if (timelimit) {
				context.trainForDuration (D, timeunit);
			} else {
				if (N > 0){
					context.train (N, taskunit);
					/* 
					context.trainAndTest(N, unit);
					*/
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
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
