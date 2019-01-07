package uk.ac.imperial.lsds.crossbow;

import uk.ac.imperial.lsds.crossbow.kernel.Dummy;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class TestMemoryPlanner {
	
	public static void main (String [] args) throws Exception {
		
		int batchSize = 128;
		
		SystemConf.getInstance().setCPU(true).setGPU(false);
		
		/* Configure dataset */
		String dataDirectory = String.format("%s/data/cifar-10/b%d/", SystemConf.getInstance().getHomeDirectory(), batchSize);
		
		Dataset dataset = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));
		
		ModelConf.getInstance ().setDataset (Phase.TRAIN, dataset);
		ModelConf.getInstance ().setBatchSize(batchSize);
		ModelConf.getInstance ().setSolverConf (new SolverConf());
		
		/* Set dataflow */
		DataflowNode head = new DataflowNode (new Operator (String.format("Dummy-%03d", 0), new Dummy ()));
		DataflowNode tail = new DataflowNode (new Operator (String.format("DummyGradient-%03d", 0), new Dummy ()).setPeer(head.getOperator()));
		
		DataflowNode next = head;
		DataflowNode prev = tail;
		
		for (int i = 1; i < 10; ++i) {
			
			DataflowNode  forward = new DataflowNode (new Operator (String.format("Dummy-%03d", i), new Dummy ()));
			DataflowNode backward = new DataflowNode (new Operator (String.format("DummyGradient-%03d", i), new Dummy ()).setPeer(forward.getOperator()));
			
			if (next == null) {
				head = forward;
				next = head;
			}
			next = next.connectTo(forward);
			backward.connectTo(prev);
			prev = backward;
		}
		
		next.connectTo(prev);
		
		SubGraph graph = new SubGraph (head);

		Dataflow [] dataflows = new Dataflow [] { new Dataflow (graph).setPhase(Phase.TRAIN), null };
		
		SystemConf.getInstance().allowMemoryReuse(true);
		
		ExecutionContext context = new ExecutionContext (dataflows);
		
		try {
		
			context.init();
			context.getDataflow(Phase.TRAIN).dump ();
			context.getDataflow(Phase.TRAIN).dumpMemoryRequirements ();
		
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
