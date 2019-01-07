package uk.ac.imperial.lsds.crossbow.dispatcher;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;
import uk.ac.imperial.lsds.crossbow.result.IResultHandler;
import uk.ac.imperial.lsds.crossbow.task.Task;
import uk.ac.imperial.lsds.crossbow.task.TaskFactory;
import uk.ac.imperial.lsds.crossbow.task.TaskQueue;

/*
 * Suppose there are two sub-graphs, A and B, wired together: A -> B.
 * When A finishes processing a batch, say b, it will pass b to B and 
 * a new task will be created.
 */
public class TransientTaskDispatcher implements ITaskDispatcher {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (TransientTaskDispatcher.class);
	
	private SubGraph graph;
	private TaskQueue queue;
	
	private IResultHandler handler;
	
	private boolean test;
	
	public TransientTaskDispatcher (SubGraph G) {
		
		graph = G;
		
		queue = graph.getDataflow().getExecutionContext().getExecutorQueue();
		handler = graph.getDataflow().getResultHandler();
		
		test = graph.getDataflow().isTest();
	}
	
	public void dispatch (Batch batch, Integer replicaId) {
		
		int taskid = batch.getId();
		
		Task task = TaskFactory.newInstance (taskid, graph, batch, handler, replicaId);
		
		task.setValidationTask(test);
			
		queue.add(task);
	}
	
	public void dispatchNext (int taskid, int bound) {
		
		throw new UnsupportedOperationException ("error: unsupported task dispatcher method call");
	}

	public VirtualCircularDataBuffer [] getBuffers () {
		
		throw new UnsupportedOperationException ("error: unsupported task dispatcher method call");
	}
}
