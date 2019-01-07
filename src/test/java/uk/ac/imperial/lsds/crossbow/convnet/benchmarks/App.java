package uk.ac.imperial.lsds.crossbow.convnet.benchmarks;

import java.io.File;
import java.io.IOException;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Dataset;
import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.LeNet;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

public class App {
	
	public static void main (String [] args) throws IOException {
		
		long startTime, dt;
		
		String mybenchmark = "alexnet"; /* "alexnet", "overfeat", "oxfordnet", "googlenet" */
		int mybatchsize = 32;
		
		int replicas = 1;
		int [] devices = new int [] { 0 };
		int wpc = replicas * devices.length * 1000000;
		
		Options options = new Options (LeNet.class.getName());
		
		options.addOption ("--N",             "Train for N units",                true,  Integer.class,       "1000");
		options.addOption ("--training-unit", "Training unit",                    true,   String.class,     "tasks");
		options.addOption ("--benchmark",     "Application benchmark model",      true,   String.class, mybenchmark);
		options.addOption ("--forward-only",  "Perform forward computation only", true,  Boolean.class,     "false");
		options.addOption ("--input-dir",     "Input data directory",             true,     File.class, "/data/crossbow/random");
		
		CommandLine commandLine = new CommandLine (options);
		
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (replicas)
			.setNumberOfGPUStreams (replicas)
			.setNumberOfGPUTaskHandlers(1)
			.setNumberOfGPUCallbackHandlers (8)
			.setDisplayInterval (100000)
			.displayAccumulatedLossValue (true)
			.queueMeasurements (true)
			.setGPUDevices(devices)
			.allowMemoryReuse(true)
			.setRandomSeed(123456789);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (mybatchsize).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.SYNCHRONOUSEAMSGD);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.1F)
			.setMomentum (0.9F)
			.setWeightDecay(0.0001F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		String benchmark = options.getOption("--benchmark").getStringValue();
		boolean backward = (! options.getOption("--forward-only").getBooleanValue());
		
		String inputDir = options.getOption("--input-dir").getStringValue();
		
		String dir = null;
		DataflowNode node = null;
		
		if (benchmark.equals("alexnet")) 
		{
			/* Input dimension: 128 x 3 x 224 x 224 */
			dir = String.format("%s/%03d-%03d-%03d-%03d/", inputDir, ModelConf.getInstance().getBatchSize(), 3, 224, 224);
			
			node = ModelBuilder.buildAlexNet(backward);
		} 
		else if (benchmark.equals("overfeat")) 
		{
			/* Input dimension: 128 x 3 x 231 x 231 */
			dir = String.format("%s/%03d-%03d-%03d-%03d/", inputDir, ModelConf.getInstance().getBatchSize(), 3, 231, 231);
			
			node = ModelBuilder.buildOverfeat(backward);
		} 
		else if (benchmark.equals("oxfordnet") || benchmark.equals("vgga")) 
		{
			/* Input is 64 x 3 x 224 x 224 */
			dir = String.format("%s/%03d-%03d-%03d-%03d/", inputDir, ModelConf.getInstance().getBatchSize(), 3, 224, 224);
			
			node = ModelBuilder.buildOxfordNet(backward);	
		} 
		else if (benchmark.equals("googlenet"))
		{
			/* Input is 128 x 3 x 224 x 224 */
			dir = String.format("%s/%03d-%03d-%03d-%03d/", inputDir, ModelConf.getInstance().getBatchSize(), 3, 224, 224);
			
			node = ModelBuilder.buildGoogleNetv1(backward);	
		} 
		else {
			System.err.println(String.format("error: invalid benchmark: %s. Valid values are: 'alexnet', 'overfeat', 'oxfordnet', 'vgga', and 'googlenet'.", benchmark));
			System.exit(1);
		}
		
		/* Load dataset(s) */
		
		Dataset dataset = new Dataset (dir + "train.metadata");
		
		ModelConf.getInstance ().setDataset (Phase.TRAIN, dataset).setDataset (Phase.CHECK, null);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		Dataflow dataflow = new Dataflow (new SubGraph (node)).setPhase(Phase.TRAIN);
		
		Dataflow [] dataflows = new Dataflow [] { dataflow, null };
		
		ExecutionContext context = new ExecutionContext (dataflows);
		
		try {
			
			context.init ();

			context.getDataflow(Phase.TRAIN).dump();
			context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
			
			context.getModel().dump();
			
			startTime = System.nanoTime ();
		
			context.train(N, unit);
			
			dt = System.nanoTime() - startTime;
			System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
			
			if (SystemConf.getInstance().queueMeasurements())
				context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();
			
			context.destroy();
		
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
