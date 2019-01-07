package uk.ac.imperial.lsds.crossbow;

import java.io.File;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.Noop;
import uk.ac.imperial.lsds.crossbow.kernel.conf.NoopConf;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class NoopApp {
	
	public static void main (String [] args) throws Exception {
		
		long startTime, dt;
		
		Options options = new Options (NoopApp.class.getName());
		
		/* Generic options */
		options.addOption ("--N",             "Train for N units", true,  Integer.class, "2");
		options.addOption ("--training-unit", "Training unit",     true,   String.class, "tasks");
		
		/* Application-specific options */
		options.addOption ("--chain", "Dataflow depth", true, Integer.class, "10");
		options.addOption ("--dataset-alias", "Dataset alias",  true,  String.class, "cifar-10");
		
		CommandLine commandLine = new CommandLine (options);
		
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setGPUDevices(0,1)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (1)
			.setNumberOfGPUStreams (1)
			.setNumberOfGPUCallbackHandlers (8)
			.setDisplayInterval (10000)
			.displayAccumulatedLossValue (true)
			.queueMeasurements (true)
			.setTaskQueueSizeLimit(32)
			.setPerformanceMonitorInterval(2000)
			.setNumberOfResultSlots(128);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (64).setWpc (10000000).setSlack (0);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.01F)
			.setMomentum (0);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		String filename = null;
		String alias = options.getOption("--dataset-alias").getStringValue ();
		
		if (alias.equalsIgnoreCase("cifar-10")) {
			filename = String.format("/data/crossbow/cifar-10/b-%03d/cifar-train.metadata", ModelConf.getInstance().getBatchSize());
		}
		else if (alias.equalsIgnoreCase("imagenet")) {
			filename = String.format("/data/crossbow/imagenet/ilsvrc2012/b-%03d/imagenet-train.metadata", ModelConf.getInstance().getBatchSize());
		}
		else {
			System.err.println(String.format("error: invalid dataset alias: %s", alias));
			System.exit(1);
		}
		/* Does the metadata file exist? */
		if (! (new File (filename)).exists()) {
			System.err.println(String.format("error: file %s not found", filename));
			System.exit(1);
		}
		
		Dataset ds1 = new Dataset (filename);
		Dataset ds2 = null;
		
		startTime = System.nanoTime ();
		
		ModelConf.getInstance().setDataset(Phase.TRAIN, ds1).setDataset(Phase.CHECK, ds2);
		
		int depth = options.getOption("--chain").getIntValue();
		
		Operator op1 = new Operator ("Noop-1", new Noop (new NoopConf().setAxis(3)));
		DataflowNode h1 = new DataflowNode (op1);
		
		/* Build chain of operators */
		if (depth > 1) {
			DataflowNode node = h1;
			for (int i = 1; i < depth; i++) {
				node = node.connectTo(new DataflowNode (new Operator (String.format("Noop-%d", (i + 1)), new Noop (new NoopConf().setAxis(3)))));
			}
		}
		
		SubGraph g1 = new SubGraph (h1);
		
		Dataflow df1 = new Dataflow (g1).setPhase(Phase.TRAIN);
		Dataflow df2 = null;
		
		ExecutionContext context = new ExecutionContext (new Dataflow [] { df1, df2 });
		
		context.init();
		context.getDataflow(Phase.TRAIN).dump();
		context.getModel().dump();
		
		context.train(N, unit);
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
