package uk.ac.imperial.lsds.crossbow.microbenchmarks.kernels;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Dataset;
import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.kernel.DataTransform;
import uk.ac.imperial.lsds.crossbow.kernel.conf.DataTransformConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class TestDataTransform {
	
	public static void main (String [] args) throws Exception {

		int batchSize = 64;
		
		int N = 1;
		TrainingUnit unit = TrainingUnit.TASKS;
		
		SystemConf.getInstance().setCPU(false).setGPU(true);
		
		/* Configure dataset */
		String dataDirectory = String.format("/data/crossbow/imagenet/ilsvrc2012/b-%d/", batchSize);
		
		Dataset dataset = new Dataset (DatasetUtils.buildPath(dataDirectory, "imagenet-train.metadata", true));
		
		ModelConf.getInstance().setDataset(Phase.TRAIN, dataset).setSolverConf(new SolverConf ());
		
		/* Set dataflow */
		
		DataTransformConf conf = new DataTransformConf ();

		conf.setMeanImageFilename("/data/crossbow/imagenet/ilsvrc2012/imagenet-train.mean");
		
		Operator transform = new Operator ("transform", new DataTransform (conf));
		DataflowNode node = new DataflowNode (transform);
		
		SubGraph graph = new SubGraph (node);
		
		Dataflow [] dataflows = new Dataflow [] { new Dataflow (graph).setPhase(Phase.TRAIN), null };
		
		ExecutionContext context = new ExecutionContext (dataflows);
		
		context.init();
		
		context.getDataflow(Phase.TRAIN).dump();
		
		try {
			context.train(N, unit);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
