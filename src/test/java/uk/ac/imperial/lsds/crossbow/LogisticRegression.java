package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.Accuracy;
import uk.ac.imperial.lsds.crossbow.kernel.GradientDescentOptimiser;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProduct;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProductGradient;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMax;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLoss;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLossGradient;
import uk.ac.imperial.lsds.crossbow.kernel.conf.AccuracyConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.MomentumMethod;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

public class LogisticRegression {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (LogisticRegression.class);
	
	public static void main (String [] args) throws Exception {
		
		long startTime, dt;
		
		Options options = new Options (LeNet.class.getName());
		
		options.addOption ("--training-unit",   "Training unit",     true,   String.class,  "epochs");
		options.addOption ("--N",             	"Train for N units", true,  Integer.class,     "100");
		options.addOption ("--target-loss",   	"Target loss",       false,   Float.class,       "0");
		
		/* Switch between xavier and constant weight initialisation */
		options.addOption ("--weight-initialiser", "Weight initialisation type", false, String.class, "gaussian");
		
		CommandLine commandLine = new CommandLine (options);
		
		int numberofreplicas = 20;
		int [] devices = new int [] { 0 };
		int wpc = numberofreplicas * devices.length * 1000000;
		
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
			.setNumberOfGPUTaskHandlers(1)
			.setDisplayInterval (1)
			.setDisplayIntervalUnit(TrainingUnit.EPOCHS)
			.displayAccumulatedLossValue (true)
			.queueMeasurements(true)
			.setTaskQueueSizeLimit(32)
			.setPerformanceMonitorInterval(1000)
			.setNumberOfResultSlots(128)
			.setGPUDevices(devices)
			.allowMemoryReuse(true)
			.setRandomSeed(123456789L);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (512).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.SYNCHRONOUSEAMSGD);
		ModelConf.getInstance ().getSolverConf ().setAlpha (0.5F).setTau (3);
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.01F)
			.setMomentum (0.0F)
			.setMomentumMethod(MomentumMethod.POLYAK);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		String dir = String.format("/data/crossbow/mnist/b-%03d/", ModelConf.getInstance().getBatchSize());
		
		Dataset ds1 = new Dataset (dir + "mnist-train.metadata");
		Dataset ds2 = new Dataset (dir +  "mnist-test.metadata");
		
		/* Load dataset(s) */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, ds1).setDataset (Phase.CHECK, ds2);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		/* Operator configuration */
		InitialiserType initialiserType = 
				InitialiserType.fromString (options.getOption("--weight-initialiser").getStringValue());
		
		if (! (initialiserType.equals(InitialiserType.CONSTANT) || initialiserType.equals(InitialiserType.GAUSSIAN ))) {
			
			System.err.println("error: weight initialiser must be neither constant or gaussian");
			System.exit(1);
		}
		
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
		
		DataflowNode             ip2 = new DataflowNode (new Operator (        "InnerProduct", new             InnerProduct (     ipConf2)));
		DataflowNode         softmax = new DataflowNode (new Operator (             "SoftMax", new                  SoftMax ( softMaxConf)));
		DataflowNode            loss = new DataflowNode (new Operator (                "Loss", new              SoftMaxLoss (    lossConf)));
		DataflowNode    lossGradient = new DataflowNode (new Operator (        "LossGradient", new      SoftMaxLossGradient (    lossConf)).setPeer (   loss.getOperator()));
		DataflowNode     ipGradient2 = new DataflowNode (new Operator ("InnerProductGradient", new     InnerProductGradient (     ipConf2)).setPeer (    ip2.getOperator()));
		DataflowNode       optimiser = new DataflowNode (new Operator (           "Optimiser", new GradientDescentOptimiser (  solverConf)));
		DataflowNode        accuracy = new DataflowNode (new Operator (            "Accuracy", new                 Accuracy (accuracyConf)));
		
		ip2.connectTo(softmax);
		softmax.connectTo(loss);
		softmax.connectTo(accuracy);
		loss.connectTo(lossGradient).connectTo(ipGradient2).connectTo(optimiser);

		/* Testing */
		
		DataflowNode             _ip2 = ip2.shallowCopy();
		DataflowNode         _softmax = softmax.shallowCopy();
		DataflowNode        _accuracy = accuracy.shallowCopy();
		
		_ip2.connectTo(_softmax).connectTo(_accuracy);
		
		SubGraph g1 = new SubGraph ( ip2);
		SubGraph g2 = new SubGraph (_ip2);
		
		Dataflow df1 = new Dataflow (g1).setPhase(Phase.TRAIN);
		Dataflow df2 = new Dataflow (g2).setPhase(Phase.CHECK);
		
		ExecutionContext context = new ExecutionContext (new Dataflow [] { df1, df2 });
		
		context.init();
		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		
		context.getModel().dump();
		
		startTime = System.nanoTime ();
		
		if (N > 0){
			try {
			    context.train(N, unit);
				/* context.trainAndTest(N, unit); */
			    /* context.trainForDuration (2, DurationUnit.MINUTES); */
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
