package uk.ac.imperial.lsds.crossbow.dlbench;

import java.io.IOException;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Dataset;
import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.LeNet;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProduct;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProductGradient;
import uk.ac.imperial.lsds.crossbow.kernel.ReLU;
import uk.ac.imperial.lsds.crossbow.kernel.ReLUGradient;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMax;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLoss;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLossGradient;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ReLUConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.types.ActivationMode;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

public class MLP {
	
	public static DataflowNode buildMultiLayerPerceptron (boolean gradient) {
		
		DataflowNode [] f = new DataflowNode [9];
		
		InnerProductConf fc1conf = new InnerProductConf ().setNumberOfOutputs (2048);
		InnerProductConf fc2conf = new InnerProductConf ().setNumberOfOutputs (4096);
		InnerProductConf fc3conf = new InnerProductConf ().setNumberOfOutputs (1024);
		InnerProductConf fc4conf = new InnerProductConf ().setNumberOfOutputs (  10);
		
		ReLUConf relu1conf = new ReLUConf ().setActivationMode (ActivationMode.SIGMOID);
		ReLUConf relu2conf = new ReLUConf ().setActivationMode (ActivationMode.SIGMOID);
		ReLUConf relu3conf = new ReLUConf ().setActivationMode (ActivationMode.SIGMOID);
		
		SoftMaxConf softmaxconf = new SoftMaxConf ();
		LossConf lossconf = new LossConf ();
		
		f[0] = new DataflowNode (new Operator ("FC-1",    new InnerProduct (    fc1conf)));
		f[1] = new DataflowNode (new Operator ("ReLU-1",  new ReLU         (  relu1conf)));
		f[2] = new DataflowNode (new Operator ("FC-2",    new InnerProduct (    fc2conf)));
		f[3] = new DataflowNode (new Operator ("ReLU-2",  new ReLU         (  relu2conf)));
		f[4] = new DataflowNode (new Operator ("FC-3",    new InnerProduct (    fc3conf)));
		f[5] = new DataflowNode (new Operator ("ReLU-3",  new ReLU         (  relu3conf)));
		f[6] = new DataflowNode (new Operator ("FC-4",    new InnerProduct (    fc4conf)));
		f[7] = new DataflowNode (new Operator ("SoftMax", new SoftMax      (softmaxconf)));
		f[8] = new DataflowNode (new Operator ("Loss",    new SoftMaxLoss  (   lossconf)));
		
		for (int i = 0; i < f.length - 1; i++)
			f[i].connectTo(f[i + 1]);
		
		if (gradient) {

			/* Gradient dataflow */
			DataflowNode [] g = new DataflowNode [8];

			g[0] = new DataflowNode (new Operator ("Loss Gradient",   new SoftMaxLossGradient  ( lossconf)).setPeer (f[8].getOperator ()));
			g[1] = new DataflowNode (new Operator ("FC-4 Gradient",   new InnerProductGradient (  fc4conf)).setPeer (f[6].getOperator ()));
			g[2] = new DataflowNode (new Operator ("ReLU-3 Gradient", new ReLUGradient         (relu3conf)).setPeer (f[5].getOperator ()));
			g[3] = new DataflowNode (new Operator ("FC-3 Gradient",   new InnerProductGradient (  fc3conf)).setPeer (f[4].getOperator ()));
			g[4] = new DataflowNode (new Operator ("ReLU-2 Gradient", new ReLUGradient         (relu2conf)).setPeer (f[3].getOperator ()));
			g[5] = new DataflowNode (new Operator ("FC-2 Gradient",   new InnerProductGradient (  fc2conf)).setPeer (f[2].getOperator ()));
			g[6] = new DataflowNode (new Operator ("ReLU-1 Gradient", new ReLUGradient         (relu1conf)).setPeer (f[1].getOperator ()));
			g[7] = new DataflowNode (new Operator ("FC-1 Gradient",   new InnerProductGradient (  fc1conf)).setPeer (f[0].getOperator ()));

			for (int i = 0; i < g.length - 1; i++)
				g[i].connectTo(g[i + 1]);

			f[8].connectTo(g[0]);
		}
		
		return f[0];
	}
	public static void main (String [] args) throws IOException {
		
		long startTime, dt;
		
		Options options = new Options (LeNet.class.getName());
		
		options.addOption ("--N",             "Train for N units",                true,  Integer.class,      "10");
		options.addOption ("--training-unit", "Training unit",                    true,   String.class,  "epochs");
		options.addOption ("--forward-only",  "Perform forward computation only", true,  Boolean.class,   "false");
		
		CommandLine commandLine = new CommandLine (options);
		
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (1)
			.setNumberOfGPUStreams (1)
			.setNumberOfGPUCallbackHandlers (8)
			.setDisplayInterval (100)
			.displayAccumulatedLossValue (true)
			.queueMeasurements (true)
			.setGPUDevices(0)
			.allowMemoryReuse(true)
			.setRandomSeed(123456789);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (64).setWpc (100000000).setSlack (0).setUpdateModel(UpdateModel.WORKER);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.1F)
			.setMomentum (0.9F)
			.setWeightDecay(0.0001F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		/* Configure dataset (i.e., MNIST) */
		String dir = String.format("%s/data/mnist/b-%03d/", SystemConf.getInstance().getHomeDirectory(), ModelConf.getInstance().getBatchSize());
		
		Dataset ds1 = new Dataset (dir + "mnist-train.metadata");
		Dataset ds2 = null;
		
		/* Load dataset(s) */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, ds1).setDataset (Phase.CHECK, ds2);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		boolean backward = (! options.getOption("--forward-only").getBooleanValue());
		
		DataflowNode node = buildMultiLayerPerceptron (backward);
		
		Dataflow dataflow = new Dataflow (new SubGraph (node)).setPhase(Phase.TRAIN);
		
		Dataflow [] dataflows = new Dataflow [] { dataflow, null };

		ExecutionContext context = new ExecutionContext (dataflows);

		try {
			
			startTime = System.nanoTime ();
			
			context.init();

			context.getDataflow(Phase.TRAIN).dump();
			context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
			context.getDataflow(Phase.TRAIN).exportDot(SystemConf.getInstance().getHomeDirectory() + "/mlp.dot");
			
			context.getModel().dump();
		
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
