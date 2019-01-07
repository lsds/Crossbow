package uk.ac.imperial.lsds.crossbow.kernel;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public abstract class Kernel implements IKernel {
	
	protected Operator operator = null;

	protected Shape outputShape = null;
	
	protected LocalVariable theInput, theOutput = null;
	
	protected KernelMemoryRequirements memoryRequirements = new KernelMemoryRequirements ();
	
	/*
	 * Temporary variables that hold the start and end pointers of
	 * the buffer returned by one of the getter methods below.
	 */
	private ThreadLocal<Integer> p = new ThreadLocal<Integer>();
	private ThreadLocal<Integer> q = new ThreadLocal<Integer>();
	
	
	public void setOperator (Operator operator) {
		
		this.operator = operator;
	}
	
	public Shape getOutputShape () {
		/*
		 * if (outputShape == null)
		 *	throw new NullPointerException("error: kernel output shape is null");
		 */
		return outputShape;
	}
	
	public DataType getOutputType () {
		
		if (theOutput == null)
			throw new NullPointerException (String.format("error: kernel output is null"));
		
		Variable [] output = theOutput.getInitialValue();
		return output[0].getType();
	}
	
	public int getOutputSize () {
		
		if (theOutput == null)
			throw new NullPointerException (String.format("error: kernel output is null"));
		
		Variable [] output = theOutput.getInitialValue();
		return output[0].capacity();
	}
	
	public KernelMemoryRequirements getKernelMemoryRequirements () {
		
		return memoryRequirements;
	}
	
	protected IDataBuffer getOperatorInput (Operator operator, Batch batch, ITask api) {
		
		/* Find upstream nodes for given operator */
		Operator [] upstreams = operator.getDataflowNode(api.getPhase()).getPreviousOperators ();
		
		if (upstreams == null) {
			/* Set start/end pointers */
			p.set(batch.getBufferStartPointer(0));
			q.set(batch.getBufferEndPointer(0));
			/* Return input examples */
			return batch.getInputBuffer(0);
		}
		else {
			/* Assert that there is only one upstream operator */
			if (upstreams.length > 1)
				throw new IllegalStateException ();
			
			Operator prev = upstreams [0];
			
			/* Assert that upstream operator has produced at least one output */
			if (batch.getOutput(prev.getId()).size() < 1)
				throw new IllegalStateException ();
			
			if (batch.getOutput(prev.getId()).size() > 1) {
				/* Find downstream nodes for upstream operator */
				Operator [] downstreams = prev.getDataflowNode(api.getPhase()).getNextOperators();
				for (int order = 0; order < downstreams.length; ++order) {
					if (downstreams[order].equals(operator)) {
						
						IDataBuffer buffer = batch.getOutput(prev.getId()).peek (order);
						if (buffer == null)
							throw new NullPointerException ();
						p.set(0);
						q.set(buffer.limit ());
						return buffer;
					}
				}
				/* Unlikely */
				throw new IllegalStateException ();
			}
			else {
				IDataBuffer buffer = batch.getOutput(prev.getId()).peek();
				p.set(0);
				q.set(buffer.limit());
				return buffer;
			}
		}
	}
	
	protected IDataBuffer getCurrentInput (Batch batch, ITask api) {
		return getOperatorInput (operator, batch, api);
	}
	
	protected IDataBuffer getPeerInput (Batch batch, ITask api) {
		
		Operator peer = operator.getPeer();
		if (peer == null)
			throw new NullPointerException(String.format("Operator %s's peer is null", operator.getName()));
		
		return getOperatorInput (peer, batch, api);
	}
	
	protected IDataBuffer getOperatorOutput (Operator op, Batch batch, ITask api) {
		if (batch.getOutput(op.getId()).size() == 1) {
			IDataBuffer buffer = batch.getOutput(op.getId()).peek();
			p.set(0);
			q.set(buffer.limit());
			return buffer;
		}
		else {
			/* 
			 * Operator `op` has produced more than one outputs. Try to find 
			 * current operator in `op`'s list of downstream nodes.
			 */
			Operator [] downstreams = op.getDataflowNode(api.getPhase()).getNextOperators();
			for (int order = 0; order < downstreams.length; ++order) {
				if (downstreams[order].equals(operator)) {
					
					IDataBuffer buffer = batch.getOutput(op.getId()).peek (order);
					if (buffer == null)
						throw new NullPointerException ();
					p.set(0);
					q.set(buffer.limit ());
					return buffer;
				}
			}
			/* Unlikely */
			throw new IllegalStateException ();
		}
	}
	
	protected IDataBuffer getCurrentOutput (Batch batch, ITask api) {
		
		DataflowNode node = operator.getDataflowNode(api.getPhase());
		
		if (! node.getOutputBufferFromElsewhere()) {
			return operator.getOutputDataBufferInstance ();
		}
		else {
			/* Reuse an output buffer */
			Operator op = node.getOutputBufferDonor ().getOperator ();
			int position = node.getOutputBufferFromPosition();
			
			IDataBuffer buffer = batch.getOutput(op.getId()).peek(position);
			if (buffer == null)
				throw new NullPointerException ();
			buffer.referenceCountGetAndIncrement ();
			p.set(0);
			q.set(buffer.limit());
			
			return buffer;
		}
	}
	
	protected IDataBuffer getPeerOutput (Batch batch, ITask api) {
		
		Operator peer = operator.getPeer();
		if (peer == null)
			throw new NullPointerException(String.format("Operator %s's peer is null", operator.getName()));
		
		/* Assert that peer operator has produced at most one output */
		if (batch.getOutput(peer.getId()).size() > 1)
			throw new IllegalStateException ();
		
		IDataBuffer buffer = batch.getOutput(peer.getId()).peek();
		p.set(0);
		q.set(buffer.limit());
		return buffer;
	}
	
	protected int getStartPointer () {
		return p.get().intValue();
	}
	
	protected int getEndPointer () {
		return q.get().intValue();
	}
}
