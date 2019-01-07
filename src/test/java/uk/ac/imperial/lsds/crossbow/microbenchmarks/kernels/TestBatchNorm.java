package uk.ac.imperial.lsds.crossbow.microbenchmarks.kernels;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Dataset;
import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.kernel.BatchNorm;
import uk.ac.imperial.lsds.crossbow.kernel.conf.BatchNormConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class TestBatchNorm {
	
	public static void main (String [] args) throws Exception {
		
		int batchSize = 128;
		
		int N = 1;
		TrainingUnit unit = TrainingUnit.TASKS;

		SystemConf.getInstance().setCPU(true).setGPU(false);
		
		/* Configure dataset */
		String dataDirectory = String.format("%s/data/cifar-10/b-%d/", SystemConf.getInstance().getHomeDirectory(), batchSize);
		
		Dataset dataset = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));
		
		ModelConf.getInstance ().setDataset (Phase.TRAIN, dataset);
		ModelConf.getInstance ().setBatchSize(batchSize);
		ModelConf.getInstance ().setSolverConf (new SolverConf());
		/* We don't want to synchronise CPU models at the moment. Afterall, this is just a unit test */
		ModelConf.getInstance ().setWpc(N + 1);
		
		/* Set dataflow */
		
		BatchNormConf conf = new BatchNormConf ();

        conf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
        conf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));
        
        Operator batchnorm = new Operator ("batchnorm", new BatchNorm (conf));
        DataflowNode node = new DataflowNode (batchnorm);
		
		SubGraph graph = new SubGraph (node);

		Dataflow [] dataflows = new Dataflow [] { new Dataflow (graph).setPhase(Phase.TRAIN), null };
		
		ExecutionContext context = new ExecutionContext (dataflows);
		
		try {
		
			context.init();
			context.getDataflow(Phase.TRAIN).dump();
			context.getModel().dump();
			context.train(N, unit);
		
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
