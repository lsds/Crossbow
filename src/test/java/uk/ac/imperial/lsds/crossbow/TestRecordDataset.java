package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.Accuracy;
import uk.ac.imperial.lsds.crossbow.kernel.Conv;
import uk.ac.imperial.lsds.crossbow.kernel.ConvGradient;
import uk.ac.imperial.lsds.crossbow.kernel.Dropout;
import uk.ac.imperial.lsds.crossbow.kernel.DropoutGradient;
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
import uk.ac.imperial.lsds.crossbow.kernel.conf.DropoutConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ReLUConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.MomentumMethod;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

public class TestRecordDataset {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (TestRecordDataset.class);
	
	public static void main (String [] args) throws Exception {
		
		long startTime, dt;
		
		Options options = new Options (TestRecordDataset.class.getName());
		
		options.addOption ("--training-unit",  "Training unit",     true,  String.class, "epochs");
		options.addOption ("--N",              "Train for N units", true, Integer.class, "1");
		options.addOption ("--data-directory", "Data directory",    true,  String.class, "/data/crossbow/imagenet/ilsvrc2012/records/");
		
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
			.setNumberOfGPUCallbackHandlers (2)
			.setNumberOfGPUTaskHandlers (2)
			.setNumberOfFileHandlers (8)
			.setMaxNumberOfMappedPartitions (1)
			.setDisplayInterval (1)
			.setDisplayIntervalUnit (TrainingUnit.EPOCHS)
			.displayAccumulatedLossValue (true)
			.setRandomSeed (123456789L)
			.queueMeasurements (true)
			.setTaskQueueSizeLimit (32)
			.setPerformanceMonitorInterval (1000)
			.setNumberOfResultSlots (128)
			.setGPUDevices (devices)
			.allowMemoryReuse (true)
			.autotuneModels (false)
			.useDirectScheduling(true);
			
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (32).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.SYNCHRONOUSEAMSGD);
		ModelConf.getInstance ().getSolverConf ().setAlpha (0.1f).setTau (3);
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.1F)
			.setMomentum (0.9F)
			.setMomentumMethod(MomentumMethod.POLYAK)
			.setWeightDecay(0.00001F)
			.setLearningRateStepUnit(TrainingUnit.EPOCHS)
			.setStepValues(new int [] { 30, 60, 80 })
			.setGamma(0.1F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		String dir = options.getOption("--data-directory").getStringValue();
		
		IDataset []  dataset = new IDataset [] { null, null };
		
		dataset [0] = new RecordDataset (DatasetUtils.buildPath(dir, "imagenet-train.metadata", true));
		dataset [1] = new RecordDataset (DatasetUtils.buildPath(dir, "imagenet-test.metadata",  true));
		
		/* Load dataset(s) */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, dataset [0]).setDataset (Phase.CHECK, dataset [1]);
		
		
		/* Create dataflows */
		
		ConvConf convConf1 = new ConvConf ();
		
		convConf1.setNumberOfOutputs (32);
		convConf1.setKernelSize (2).setKernelHeight (5).setKernelWidth (5);
		
		convConf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		convConf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));

		ReLUConf reluConf1 = new ReLUConf ();
		
		PoolConf poolConf1 = new PoolConf ().setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		ConvConf convConf2 = new ConvConf ();
		
		convConf2.setNumberOfOutputs (64);
		convConf2.setKernelSize (2).setKernelHeight (5).setKernelWidth (5);
		
		convConf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		convConf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
		
		ReLUConf reluConf2 = new ReLUConf ();

		PoolConf poolConf2 = new PoolConf ().setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		InnerProductConf ipConf1 = new InnerProductConf();
		
		ipConf1.setNumberOfOutputs (1024);
		
		ipConf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		ipConf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
				
		ReLUConf reluConf3 = new ReLUConf ();
		
		DropoutConf dropoutConf = new DropoutConf ().setRatio(0.4F);
		
		InnerProductConf ipConf2 = new InnerProductConf ();
		
		ipConf2.setNumberOfOutputs (1000);
		
		ipConf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		ipConf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1));
				
		SoftMaxConf softMaxConf = new SoftMaxConf ();
		LossConf lossConf = new LossConf ();
		AccuracyConf accuracyConf = new AccuracyConf ();
		
		SolverConf solverConf = ModelConf.getInstance().getSolverConf();
		
		DataflowNode           conv1 = new DataflowNode (new Operator (                "Conv", new                     Conv (   convConf1)));
		DataflowNode           relu1 = new DataflowNode (new Operator (                "ReLU", new                     ReLU (   reluConf1)));
		DataflowNode           pool1 = new DataflowNode (new Operator (                "Pool", new                     Pool (   poolConf1)));
		DataflowNode           conv2 = new DataflowNode (new Operator (                "Conv", new                     Conv (   convConf2)));
		DataflowNode           relu2 = new DataflowNode (new Operator (                "ReLU", new                     ReLU (   reluConf2)));
		DataflowNode           pool2 = new DataflowNode (new Operator (                "Pool", new                     Pool (   poolConf2)));
		DataflowNode             ip1 = new DataflowNode (new Operator (        "InnerProduct", new             InnerProduct (     ipConf1)));
		DataflowNode           relu3 = new DataflowNode (new Operator (                "ReLU", new                     ReLU (   reluConf3)));
		DataflowNode         dropout = new DataflowNode (new Operator (             "Dropout", new                  Dropout ( dropoutConf)));
		DataflowNode             ip2 = new DataflowNode (new Operator (        "InnerProduct", new             InnerProduct (     ipConf2)));
		DataflowNode         softmax = new DataflowNode (new Operator (             "SoftMax", new                  SoftMax ( softMaxConf)));
		DataflowNode            loss = new DataflowNode (new Operator (                "Loss", new              SoftMaxLoss (    lossConf)));
		DataflowNode    lossGradient = new DataflowNode (new Operator (        "LossGradient", new      SoftMaxLossGradient (    lossConf)).setPeer (   loss.getOperator()));
		DataflowNode     ipGradient2 = new DataflowNode (new Operator ("InnerProductGradient", new     InnerProductGradient (     ipConf2)).setPeer (    ip2.getOperator()));
		DataflowNode dropoutGradient = new DataflowNode (new Operator (     "DropoutGradient", new          DropoutGradient ( dropoutConf)).setPeer (dropout.getOperator()));
		DataflowNode   reluGradient3 = new DataflowNode (new Operator (        "ReLUGradient", new             ReLUGradient (   reluConf3)).setPeer (  relu3.getOperator()));
		DataflowNode     ipGradient1 = new DataflowNode (new Operator ("InnerProductGradient", new     InnerProductGradient (     ipConf1)).setPeer (    ip1.getOperator()));
		DataflowNode   poolGradient2 = new DataflowNode (new Operator (        "PoolGradient", new             PoolGradient (   poolConf2)).setPeer (  pool2.getOperator()));
		DataflowNode   reluGradient2 = new DataflowNode (new Operator (        "ReLUGradient", new             ReLUGradient (   reluConf2)).setPeer (  relu2.getOperator()));
		DataflowNode   convGradient2 = new DataflowNode (new Operator (        "ConvGradient", new             ConvGradient (   convConf2)).setPeer (  conv2.getOperator()));
		DataflowNode   poolGradient1 = new DataflowNode (new Operator (        "PoolGradient", new             PoolGradient (   poolConf1)).setPeer (  pool1.getOperator()));
		DataflowNode   reluGradient1 = new DataflowNode (new Operator (        "ReLUGradient", new             ReLUGradient (   reluConf1)).setPeer (  relu1.getOperator()));
		DataflowNode   convGradient1 = new DataflowNode (new Operator (        "ConvGradient", new             ConvGradient (   convConf1)).setPeer (  conv1.getOperator()));
		DataflowNode       optimiser = new DataflowNode (new Operator (           "Optimiser", new GradientDescentOptimiser (  solverConf)));
		DataflowNode        accuracy = new DataflowNode (new Operator (            "Accuracy", new                 Accuracy (accuracyConf)));
		
		conv1.connectTo(relu1).connectTo(pool1).connectTo(conv2).connectTo(relu2).connectTo(pool2).connectTo(ip1).connectTo(relu3).connectTo(dropout).connectTo(ip2).connectTo(softmax);
		softmax.connectTo(loss);
		softmax.connectTo(accuracy);
		loss.connectTo(lossGradient).connectTo(ipGradient2).connectTo(dropoutGradient).connectTo(reluGradient3).connectTo(ipGradient1).connectTo(poolGradient2).connectTo(reluGradient2).connectTo(convGradient2).connectTo(poolGradient1).connectTo(reluGradient1).connectTo(convGradient1).connectTo(optimiser);

		/* Testing */
		
		DataflowNode           _conv1 = conv1.shallowCopy();
		DataflowNode           _relu1 = relu1.shallowCopy();
		DataflowNode           _pool1 = pool1.shallowCopy();
		DataflowNode           _conv2 = conv2.shallowCopy();
		DataflowNode           _relu2 = relu2.shallowCopy();
		DataflowNode           _pool2 = pool2.shallowCopy();
		DataflowNode             _ip1 = ip1.shallowCopy();
		DataflowNode           _relu3 = relu3.shallowCopy();
		DataflowNode             _ip2 = ip2.shallowCopy();
		DataflowNode         _softmax = softmax.shallowCopy();
		DataflowNode        _accuracy = accuracy.shallowCopy();
		
		_conv1.connectTo(_relu1).connectTo(_pool1).connectTo(_conv2).connectTo(_relu2).connectTo(_pool2).connectTo(_ip1).connectTo(_relu3).connectTo(_ip2).connectTo(_softmax).connectTo(_accuracy);
		
		SubGraph g1 = new SubGraph ( conv1);
		SubGraph g2 = new SubGraph (_conv1);
		
		Dataflow df1 = new Dataflow (g1).setPhase(Phase.TRAIN);
		Dataflow df2 = new Dataflow (g2).setPhase(Phase.CHECK);
		
		ExecutionContext context = new ExecutionContext (new Dataflow [] { df1, df2 });
		
		context.init();
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		
		context.getModel().dump();
		
		startTime = System.nanoTime ();
		
		if (N > 0){
			try {
				context.trainAndTest(N, unit);
			} catch (Exception e) {
				e.printStackTrace();
			}
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
