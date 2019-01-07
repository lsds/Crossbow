package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.kernel.MatFact;
import uk.ac.imperial.lsds.crossbow.kernel.conf.MatFactConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class MatrixFactorisation {
	
	private final static Logger log = LogManager.getLogger (MatrixFactorisation.class);
	
	public static final String usage = "usage: MatrixFactorisation";
	
	public static void main (String [] args) throws Exception {
		
		/* Parse command line arguments */
		int i, j;
		
		boolean cpu = false, gpu = true;
		// boolean cpu = true, gpu = false;
		
		int __number_of_workers = 1;
		
		int __number_of_cpu_model_replicas = 1;
		int __number_of_gpu_model_replicas = 1;
		
		int __number_of_gpu_streams = 1;
		
		int __readers_per_model = 1;
		
		int __batch_size = 4096;
		
		int __numBatchesPerEpoch = 637;
		
		SystemConf.getInstance().setNumberOfResultSlots(32768);
		
		int __N = 10;
		TrainingUnit __unit = TrainingUnit.EPOCHS;
		
		int __wpc = 637; // 2267; // 10000000;
		int __slack = 0;
		
		long startTime, dt;
		
		for (i = 0; i < args.length; ) {
			if ((j = i + 1) == args.length) {
				System.err.println(usage);
				System.exit(1);
			}
			if (args[i].equals("--cpu")) {
				cpu = Boolean.parseBoolean(args[j]);
			} else
			if (args[i].equals("--gpu")) {
				gpu = Boolean.parseBoolean(args[j]);
			} else
			if (args[i].equals("--number-of-workers")) {
				__number_of_workers = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--number-of-cpu-model-replicas")) {
				__number_of_cpu_model_replicas =  Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--number-of-gpu-model-replicas")) {
				__number_of_gpu_model_replicas =  Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--readers-per-model")) {
				__readers_per_model = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--number-of-gpu-streams")) {
				__number_of_gpu_streams = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--unit")) {
				__unit = TrainingUnit.fromString(args[j]);
			} else
			if (args[i].equals("--N")) {
				__N = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--wpc")) {
				__wpc = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--slack")) {
				__slack = Integer.parseInt(args[j]);
			} else {
				System.err.println(String.format("error: unknown flag %s %s", args[i], args[j]));
				System.exit(1);
			}
			i = j + 1;
		}
		
		startTime = System.nanoTime();
		
		String dir = SystemConf.getInstance().getHomeDirectory() + "/data/ratings/";
		Dataset ds1 = new Dataset (dir + "ratings.metadata");
		
		int L        =      10; /* Number of latent factors */
		int U        =    5000; /* Number of users */
		int D        =    5000; /* Number of items */
		float lambda =    0.1f;
		float rate   = 0.0001f;
		
		ModelConf.getInstance().setBatchSize(__batch_size).setDataset(Phase.TRAIN, ds1);
		
		int [] tasksize = ModelConf.getInstance().getTaskSize();
		
		log.info(String.format("%d examples/task %d (%d) bytes/task %d tasks/epoch",
				ModelConf.getInstance().getBatchSize(), tasksize[0], tasksize[1], ModelConf.getInstance().numberOfTasksPerEpoch()));
		
		MatFactConf conf = new MatFactConf()
			.setModelVariableInitialiser(new InitialiserConf())
			.setNumberOfLatentVariables(L)
			.setNumberOfRows(U)
			.setNumberOfColumns(D)
			.setLambda(lambda)
			.setLearningRateEta0(rate);
		
		Operator op1 = new Operator ("MatFact", new MatFact(conf));
		
		DataflowNode h1 = new DataflowNode (op1);
		
		SubGraph g1 = new SubGraph (h1);
		
		Dataflow df1 = new Dataflow (g1).setPhase(Phase.TRAIN);

		SystemConf.getInstance()
			.setCPU (cpu)
			.setGPU (gpu)
			.setNumberOfWorkerThreads (__number_of_workers)
			.setNumberOfCPUModelReplicas (__number_of_cpu_model_replicas)
			.setNumberOfReadersPerModel (__readers_per_model)
			.setNumberOfGPUModelReplicas (__number_of_gpu_model_replicas)
			.setNumberOfGPUStreams(__number_of_gpu_streams)
			.displayAccumulatedLossValue(true)
			.setDisplayInterval(__numBatchesPerEpoch)
			.queueMeasurements(true);
		
		int wpc = __wpc;
		// int wpc = Math.round(__wpc * ModelConf.getInstance().numberOfTasksPerEpoch());
		if (! (wpc > 0))
			throw new IllegalArgumentException();
		log.info(String.format("Synchronise every %d tasks", wpc));

		ModelConf.getInstance()
			// .setUnit (__unit)
			.setWpc  (wpc)
			.setSlack (__slack);
		
		ExecutionContext context = new ExecutionContext (new Dataflow [] { df1, null });
		
		context.init();
		context.getDataflow(Phase.TRAIN).dump();
		context.getModel().dump();
		
		context.train(__N, __unit);
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
		
		if (SystemConf.getInstance().queueMeasurements())
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();
		
		context.destroy();

		System.out.println("Bye.");
		System.exit(0);
	}
}
