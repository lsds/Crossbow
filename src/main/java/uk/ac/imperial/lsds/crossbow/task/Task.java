package uk.ac.imperial.lsds.crossbow.task;

import java.util.concurrent.atomic.AtomicMarkableReference;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.BatchFactory;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.result.IResultHandler;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Task extends AbstractTask {
	
	private SubGraph graph;
	private Batch batch;
	private IResultHandler handler;
	
	public Task () {
		this(0, null, null, null, null);
	}
	
	public Task (int taskId, SubGraph graph, Batch batch, IResultHandler handler, Integer replicaId) {
		
		next = new AtomicMarkableReference<AbstractTask>(null, false);
		set (taskId, graph, batch, handler, replicaId);
	}
	
	public void set (int taskId, SubGraph graph, Batch batch, IResultHandler handler, Integer replicaId) {
		
		this.taskId = taskId;
		
		this.graph = graph;
		this.batch = batch;
		this.handler = handler;
		
		this.replicaId = replicaId;
		
		if (graph == null) {
			graphId = Integer.MAX_VALUE;
			access = ModelAccess.NA;
		} else {
			graphId = graph.getId();
			access = graph.getModelAccessType();
		}
		
		if (batch != null)
			lowerBound = batch.getLowerBound();
		
		this.next.set(null, false);
	}

	@Override
	public int run () {
		graph.process(batch, replicaId, this /* API */, GPU);
		SubGraph next = graph.getNext();
		if (next != null) {
			next.getTaskDispatcher().dispatch(batch, replicaId);
		} else {
			if (! GPU) {
				handler.setSlot(taskId, batch.getFreeOffsets(), batch.getLoss(), batch.getAccuracy(), batch.getModelGradient(), GPU);
			}
			/* Free batch */
			BatchFactory.free (batch);
			/* Release model replica id */
			if (! GPU) {
				/*
				 * We cannot release the model replica of a GPU task
				 * because the task may not have finished  execution
				 * due to pipelining.
				 */
				graph.getDataflow().getExecutionContext().getModelManager().release(replicaId);
			}
		}
		return 0;
	}

	@Override
	public void free () {
		TaskFactory.free(this);
	}
	
	public void outputBatchResult (Batch batch) {
		this.batch = batch;
	}
	
	public Operator getPrevious (Operator p) {
		
		if (p == null)
			throw new NullPointerException ("error: operator is null");
		
		DataflowNode next = graph.getDataflowNode();
		while (next != null) {
			Operator q = next.getOperator();
			if (p == q) {
				DataflowNode node = next.getPreviousInTopology();
				if (node != null)  
					return node.getOperator();
				else
					return null;
			}
			next = next.getNextInTopology();
		}
		
		throw new NullPointerException (String.format("error: operator %s not found in %s", p.getName(), graph.getName()));
	}

	public boolean isMostDownstream (Operator p) {
		
		if (p == null)
			throw new NullPointerException ("error: operator is null");
		
		DataflowNode next = graph.getDataflowNode();
		while (next != null) {
			Operator q = next.getOperator();
			if (p == q) {
				/* Node found */
				return (next.getNextInTopology() == null);
			}
			next = next.getNextInTopology();
		}
		
		throw new NullPointerException (String.format("error: operator %s not found in %s", p.getName(), graph.getName()));
	}
	
	public boolean isMostUpstream (Operator p) {
		
		if (p == null)
			throw new NullPointerException ("error: operator is null");
		
		DataflowNode next = graph.getDataflowNode();
		while (next != null) {
			Operator q = next.getOperator();
			if (p == q) {
				/* Node found */
				return ((next.getPreviousInTopology() == null) || 
						(next.getPreviousInTopology().getOperator().getKernel().isDataTransformationKernel()));
			}
			next = next.getNextInTopology();
		}
		
		throw new NullPointerException (String.format("error: operator %s not found in %s", p.getName(), graph.getName()));
	}
}
