package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.Accuracy;
import uk.ac.imperial.lsds.crossbow.kernel.BatchNorm;
import uk.ac.imperial.lsds.crossbow.kernel.BatchNormGradient;
//import uk.ac.imperial.lsds.crossbow.kernel.Conv;
//import uk.ac.imperial.lsds.crossbow.kernel.ConvGradient;
//import uk.ac.imperial.lsds.crossbow.kernel.Dropout;
//import uk.ac.imperial.lsds.crossbow.kernel.DropoutGradient;
import uk.ac.imperial.lsds.crossbow.kernel.GradientDescentOptimiser;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProduct;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProductGradient;
//import uk.ac.imperial.lsds.crossbow.kernel.Pool;
//import uk.ac.imperial.lsds.crossbow.kernel.PoolGradient;
//import uk.ac.imperial.lsds.crossbow.kernel.ReLU;
//import uk.ac.imperial.lsds.crossbow.kernel.ReLUGradient;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMax;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLoss;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLossGradient;
import uk.ac.imperial.lsds.crossbow.kernel.conf.AccuracyConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.BatchNormConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConvConf;
//import uk.ac.imperial.lsds.crossbow.kernel.conf.DropoutConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
//import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
//import uk.ac.imperial.lsds.crossbow.kernel.conf.ReLUConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

public class LeNetBatchNorm {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (LeNetBatchNorm.class);
	
	public static void main (String [] args) throws Exception {
		
		long startTime, dt;
		
		Options options = new Options (LeNet.class.getName());
		
		options.addOption ("--training-unit",   "Training unit",     true,   String.class,  "epochs");
		options.addOption ("--N",             	"Train for N units", true,  Integer.class,    "100");
		options.addOption ("--target-loss",   	"Target loss",       false,   Float.class,     "0");
		
		/* Switch between xavier and constant weight initialisation */
		options.addOption ("--weight-initialiser", "Weight initialisation type", false, String.class, "gaussian");
		
		CommandLine commandLine = new CommandLine (options);
		
		int numberofreplicas = 1;
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (numberofreplicas) /* wpc too !*/
			.setNumberOfGPUStreams (numberofreplicas)
			.setNumberOfGPUCallbackHandlers (8)
			.setDisplayInterval (1)
			.setDisplayIntervalUnit (TrainingUnit.EPOCHS)
			.displayAccumulatedLossValue (true)
			.queueMeasurements (true)
			.setGPUDevices(0);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (64).setWpc (1).setSlack (0).setUpdateModel(UpdateModel.DEFAULT);
		ModelConf.getInstance ().getSolverConf (); // .setAlpha (0.5F).setTau (3);
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.01F)
			.setMomentum (0.9F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		String dir = String.format("%s/data/mnist/b-%03d/", SystemConf.getInstance().getHomeDirectory(), ModelConf.getInstance().getBatchSize());
		
		Dataset ds1 = new Dataset (dir + "mnist-train.metadata");
		Dataset ds2 = new Dataset (dir +  "mnist-test.metadata");
		
		/* Load dataset(s) */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, ds1).setDataset (Phase.CHECK, ds2);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
//		int [] tasksize = ModelConf.getInstance().getTaskSize();
//		log.info(String.format("%d examples/task %d (%d) bytes/task %d tasks/epoch", 
//				ModelConf.getInstance().getBatchSize(), tasksize[0], tasksize[1], ModelConf.getInstance().numberOfTasksPerEpoch()));
//		
//		if (! (wpc > 0))
//			throw new IllegalArgumentException();
//		log.info(String.format("Synchronise every %d tasks", wpc));
//		
//		ModelConf.getInstance().setWpc(wpc).setSlack(0).setTestInterval(ModelConf.getInstance().numberOfTasksPerEpoch());
		
		/* Operator configuration */
		InitialiserType initialiserType = 
				InitialiserType.fromString (options.getOption("--weight-initialiser").getStringValue());
		
		if (! (initialiserType.equals(InitialiserType.CONSTANT) || initialiserType.equals(InitialiserType.GAUSSIAN ))) {
			
			System.err.println("error: weight initialiser must be neither constant or gaussian");
			System.exit(1);
		}
		
		ConvConf convConf1 = new ConvConf ();
		
		convConf1.setNumberOfOutputs (32);
		convConf1.setKernelSize (2).setKernelHeight (5).setKernelWidth (5);
		
		if (initialiserType.equals(InitialiserType.CONSTANT)) {
			convConf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
			convConf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
		} else {
			convConf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.1F));
			convConf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		}
		
		BatchNormConf normConf = new BatchNormConf ();
		normConf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));
		
//		PoolConf poolConf1 = new PoolConf ().setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		ConvConf convConf2 = new ConvConf ();
		
		convConf2.setNumberOfOutputs (64);
		convConf2.setKernelSize (2).setKernelHeight (5).setKernelWidth (5);
		
		if (initialiserType.equals(InitialiserType.CONSTANT)) {
			convConf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
			convConf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
		} else {
			convConf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.1F));
			convConf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.1F));
		}

//		PoolConf poolConf2 = new PoolConf ().setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		InnerProductConf ipConf1 = new InnerProductConf();
		
		ipConf1.setNumberOfOutputs (512);
		
		if (initialiserType.equals(InitialiserType.CONSTANT)) {
			ipConf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
			ipConf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
		} else {
			ipConf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.1F));
			ipConf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.1F));
		}
				
//		ReLUConf reluConf = new ReLUConf ();
//		
//		DropoutConf dropoutConf = new DropoutConf ().setRatio(0.5F);
		
		InnerProductConf ipConf2 = new InnerProductConf ();
		
		ipConf2.setNumberOfOutputs (10);
		
		if (initialiserType.equals(InitialiserType.CONSTANT)) {
			ipConf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
			ipConf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
		} else {
			ipConf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.1F));
			ipConf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.1F));
		}
				
		SoftMaxConf softMaxConf = new SoftMaxConf ();
		LossConf lossConf = new LossConf ();
		AccuracyConf accuracyConf = new AccuracyConf ();
		
		SolverConf solverConf = ModelConf.getInstance().getSolverConf();
		
//		DataflowNode           conv1 = new DataflowNode (new Operator (                "Conv", new                     Conv (   convConf1)));
		DataflowNode           batchnorm = new DataflowNode (new Operator (                "BatchNorm", new                     BatchNorm (   normConf)));
		
//		DataflowNode           pool1 = new DataflowNode (new Operator (                "Pool", new                     Pool (   poolConf1)));
//		DataflowNode           conv2 = new DataflowNode (new Operator (                "Conv", new                     Conv (   convConf2)));
//		DataflowNode           pool2 = new DataflowNode (new Operator (                "Pool", new                     Pool (   poolConf2)));
//		DataflowNode             ip1 = new DataflowNode (new Operator (        "InnerProduct", new             InnerProduct (     ipConf1)));
//		DataflowNode            relu = new DataflowNode (new Operator (                "ReLU", new                     ReLU (    reluConf)));
//		DataflowNode         dropout = new DataflowNode (new Operator (             "Dropout", new                  Dropout ( dropoutConf)));
		DataflowNode             ip2 = new DataflowNode (new Operator (        "InnerProduct", new             InnerProduct (     ipConf2)));
		DataflowNode         softmax = new DataflowNode (new Operator (             "SoftMax", new                  SoftMax ( softMaxConf)));
		DataflowNode            loss = new DataflowNode (new Operator (                "Loss", new              SoftMaxLoss (    lossConf)));
		DataflowNode    lossGradient = new DataflowNode (new Operator (        "LossGradient", new      SoftMaxLossGradient (    lossConf)).setPeer (   loss.getOperator()));
		DataflowNode     ipGradient2 = new DataflowNode (new Operator ("InnerProductGradient", new     InnerProductGradient (     ipConf2)).setPeer (    ip2.getOperator()));
//		DataflowNode dropoutGradient = new DataflowNode (new Operator (     "DropoutGradient", new          DropoutGradient ( dropoutConf)).setPeer (dropout.getOperator()));
//		DataflowNode    reluGradient = new DataflowNode (new Operator (        "ReLUGradient", new             ReLUGradient (    reluConf)).setPeer (   relu.getOperator()));
//		DataflowNode     ipGradient1 = new DataflowNode (new Operator ("InnerProductGradient", new     InnerProductGradient (     ipConf1)).setPeer (    ip1.getOperator()));
//		DataflowNode   poolGradient2 = new DataflowNode (new Operator (        "PoolGradient", new             PoolGradient (   poolConf2)).setPeer (  pool2.getOperator()));
//		DataflowNode   convGradient2 = new DataflowNode (new Operator (        "ConvGradient", new             ConvGradient (   convConf2)).setPeer (  conv2.getOperator()));
//		DataflowNode   poolGradient1 = new DataflowNode (new Operator (        "PoolGradient", new             PoolGradient (   poolConf1)).setPeer (  pool1.getOperator()));
		DataflowNode   batchNormGradient = new DataflowNode (new Operator (        "BatchNormGradient", new             BatchNormGradient (   normConf)).setPeer (  batchnorm.getOperator()));
		
//		DataflowNode   convGradient1 = new DataflowNode (new Operator (        "ConvGradient", new             ConvGradient (   convConf1)).setPeer (  conv1.getOperator()));
		DataflowNode       optimiser = new DataflowNode (new Operator (           "Optimiser", new GradientDescentOptimiser (  solverConf)));
		DataflowNode        accuracy = new DataflowNode (new Operator (            "Accuracy", new                 Accuracy (accuracyConf)));
		
//		conv1.connectTo(batchnorm).connectTo(pool1).connectTo(conv2).connectTo(pool2).connectTo(ip1).connectTo(relu).connectTo(dropout).connectTo(ip2).connectTo(softmax);
//		softmax.connectTo(loss);
//		softmax.connectTo(accuracy);
//		loss.connectTo(lossGradient).connectTo(ipGradient2).connectTo(dropoutGradient).connectTo(reluGradient).connectTo(ipGradient1).connectTo(poolGradient2).connectTo(convGradient2).connectTo(poolGradient1).connectTo(batchNormGradient).connectTo(convGradient1).connectTo(optimiser);

		batchnorm.connectTo(ip2).connectTo(softmax);
		softmax.connectTo(loss);
		softmax.connectTo(accuracy);
		
		loss.connectTo(lossGradient).connectTo(ipGradient2).connectTo(batchNormGradient).connectTo(optimiser);
		
		/* Testing */
		
//		DataflowNode           _conv1 = conv1.shallowCopy();
		DataflowNode           _batchnorm = batchnorm.shallowCopy();
//		DataflowNode           _pool1 = pool1.shallowCopy();
//		DataflowNode           _conv2 = conv2.shallowCopy();
//		DataflowNode           _pool2 = pool2.shallowCopy();
//		DataflowNode             _ip1 = ip1.shallowCopy();
//		DataflowNode            _relu = relu.shallowCopy();
		DataflowNode             _ip2 = ip2.shallowCopy();
		DataflowNode         _softmax = softmax.shallowCopy();
		DataflowNode        _accuracy = accuracy.shallowCopy();
		
//		_conv1.connectTo(_batchnorm).connectTo(_pool1).connectTo(_conv2).connectTo(_pool2).connectTo(_ip1).connectTo(_relu).connectTo(_ip2).connectTo(_softmax).connectTo(_accuracy);
		
		_batchnorm.connectTo(_ip2).connectTo(_softmax).connectTo(_accuracy);
		
//		SubGraph g1 = new SubGraph ( conv1);
//		SubGraph g2 = new SubGraph (_conv1);
		
		SubGraph g1 = new SubGraph ( batchnorm);
		SubGraph g2 = new SubGraph (_batchnorm);
		
		Dataflow df1 = new Dataflow (g1).setPhase(Phase.TRAIN);
		Dataflow df2 = new Dataflow (g2).setPhase(Phase.CHECK);
		
		ExecutionContext context = new ExecutionContext (new Dataflow [] { df1, df2 });
		
		context.init();
		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		context.getModel().dump();
		
		startTime = System.nanoTime ();
		
		try {
		    // context.train(N, unit);
			// context.test();
			context.trainAndTest(N, unit);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
		
		if (SystemConf.getInstance().queueMeasurements()) {
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();
			context.getDataflow(Phase.CHECK).getResultHandler().getMeasurementQueue().dump();
		}
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
