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

public class TestDynamics {

	public static final String usage = "usage: TestDynamics";

	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (TestDynamics.class);
	
	public static void main (String [] args) throws Exception {

		long startTime, dt;
		
		Options options = new Options (TestDynamics.class.getName());
		
		options.addOption ("--training-unit", "Training unit",     true,  String.class, "epochs");
		options.addOption ("--N",             "Train for N units", true, Integer.class,   "10000"); // "128");
		
		CommandLine commandLine = new CommandLine (options);
		
		int numberofreplicas = 4;
		int [] devices = new int [] { 0,1,2,3 };
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
			.setNumberOfGPUTaskHandlers (4)
			.setMaxNumberOfMappedPartitions (1)
			.setDisplayInterval (10000)
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
		ModelConf.getInstance ().setBatchSize (16).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.WORKER);
		ModelConf.getInstance ().getSolverConf ().setAlpha (0.1f).setTau (3);
		ModelConf.getInstance ().setTestInterval (8).setTestIntervalUnit (TrainingUnit.TASKS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.1F)
			.setMomentum (0.0F)
			.setMomentumMethod(MomentumMethod.POLYAK)
			.setWeightDecay(0.00004F)
			.setLearningRateStepUnit(TrainingUnit.EPOCHS)
			.setStepValues(new int [] { 82, 122 })
			.setGamma(0.1F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int __N = options.getOption("--N").getIntValue ();
		TrainingUnit taskunit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		/* Build dataflows for both training and testing phases */
        SubGraph [] graphs = new SubGraph [2];
        
        InnerProductConf ipConf = new InnerProductConf ().setNumberOfOutputs (10);
        
        ipConf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd  (0.1F));
        ipConf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.1F));
        
        SoftMaxConf softMaxConf = new SoftMaxConf ();
        
        LossConf lossConf = new LossConf ();
		
        SolverConf solverConf = ModelConf.getInstance().getSolverConf ();
        
        AccuracyConf accuracyConf = new AccuracyConf ();

        DataflowNode           ip = new DataflowNode (new Operator (        "InnerProduct", new             InnerProduct (      ipConf)));
        DataflowNode      softmax = new DataflowNode (new Operator (             "SoftMax", new                  SoftMax ( softMaxConf)));
        DataflowNode         loss = new DataflowNode (new Operator (                "Loss", new              SoftMaxLoss (    lossConf)));
        DataflowNode lossGradient = new DataflowNode (new Operator (        "LossGradient", new      SoftMaxLossGradient (    lossConf)).setPeer (loss.getOperator()));
        DataflowNode   ipGradient = new DataflowNode (new Operator ("InnerProductGradient", new     InnerProductGradient (      ipConf)).setPeer (  ip.getOperator()));
        DataflowNode    optimiser = new DataflowNode (new Operator (           "Optimiser", new GradientDescentOptimiser (  solverConf)));
        DataflowNode     accuracy = new DataflowNode (new Operator (            "Accuracy", new                 Accuracy (accuracyConf)));
		
        DataflowNode          _ip =       ip.shallowCopy();
        DataflowNode     _softmax =  softmax.shallowCopy();
        DataflowNode    _accuracy = accuracy.shallowCopy();
        
        ip.connectTo(softmax).connectTo(loss);
        softmax.connectTo(accuracy);
        loss.connectTo(lossGradient).connectTo(ipGradient).connectTo(optimiser);
        
        _ip.connectTo(_softmax).connectTo(_accuracy);
        
        graphs[0] = new SubGraph ( ip);
        graphs[1] = new SubGraph (_ip);
        
		/* Set dataflow */

		Dataflow [] dataflows = new Dataflow [2];
		
        dataflows [0] = new Dataflow (graphs[0]).setPhase(Phase.TRAIN);
        dataflows [1] = null; // new Dataflow (graphs[1]).setPhase(Phase.CHECK);
        		
		/* Create dataset */
		IDataset [] dataset = new IDataset [] { null, null };
		
        String dataDirectory = String.format("/data/crossbow/cifar-10/padded/b-%03d/", ModelConf.getInstance().getBatchSize());
			
		dataset [0] = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));
		dataset [1] = null; // new Dataset (DatasetUtils.buildPath(dataDirectory,  "cifar-test.metadata", true));
        
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
			if (__N > 0){
					
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

