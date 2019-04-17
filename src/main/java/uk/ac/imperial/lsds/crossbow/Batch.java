package uk.ac.imperial.lsds.crossbow;

import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.ModelGradient;
import uk.ac.imperial.lsds.crossbow.utils.CrossbowLinkedList;

public class Batch {
	
	private int id;
	private int bound;
	
	private IDataBuffer [] inputs;
	
	private int [] tasksize;
	
	private int [] start, end; /* Start and end pointers */
	private long [] free;
	
	private CrossbowLinkedList<IDataBuffer> [] outputs = null;
	
	private ModelGradient gradient;
	
	private boolean initialised = false;
	
	private float loss;
	private float accuracy;
	
	public Batch () {
		this (0, 0, null, null, null, null, null, 0);
	}
	
	@SuppressWarnings("unchecked")
	public Batch (int id, int bound, IDataBuffer [] inputs, int [] tasksize, long [] start, long [] end, long [] free, int numberOfOperators) {
		
		this.id = id;
		this.bound = bound;
		
		this.inputs = new IDataBuffer [2];
		for (int i = 0; i < 2; ++i)
			this.inputs[i] = inputs[i];
		
		this.tasksize = new int [2];
		for (int i = 0; i < 2; ++i)
			if (tasksize != null)
				this.tasksize[i] = tasksize[i];
			else
				this.tasksize[i] = 0;
		
		this.start = new int [2];
		for (int i = 0; i < 2; ++i)
			if (start != null)
				this.start[i] = (int) start[i];
			else
				this.start[i] = 0;
		
		this.end = new int [2];
		for (int i = 0; i < 2; ++i)
			if (end != null)
				this.end[i] = (int) end[i];
			else
				this.end[i] = 0;
		
		this.free = new long [2];
		for (int i = 0; i < 2; ++i)
			if (free != null)
				this.free[i] = free[i];
			else
				this.free[i] = 0L;
		
		outputs = new CrossbowLinkedList [numberOfOperators];
		for (int i = 0; i < numberOfOperators; ++i)
			outputs[i] = new CrossbowLinkedList<IDataBuffer> ();
		
		initialised = true;
		
		loss = 0;
		accuracy = 0;
	}
	
	public void set (int id, int bound, IDataBuffer [] inputs, int [] tasksize, long [] start, long [] end, long [] free) {
		
		if (isInitialised ())
			throw new IllegalStateException("error: batch is already initialised");
		
		this.id = id;
		this.bound = bound;
		
		for (int i = 0; i < 2; ++i) {
			this.inputs[i]   =         inputs[i];
			this.tasksize[i] =       tasksize[i];
			this.start[i]    = (int)    start[i];
			this.end[i]      = (int)      end[i];
			this.free[i]     =           free[i];
		}
		
		for (int i = 0; i < outputs.length; ++i)
			if (! outputs[i].isEmpty())
				throw new IllegalStateException("error: batch has not been cleared");
		
		/*
		for (int i = 0; i < myoutputs.length; ++i)
			myoutputs [i] = null;
		*/
		this.gradient = null;
		
		initialised = true;
		
		loss = 0;
		accuracy = 0;
	}
	
	public int getId () {
		return id;
	}
	
	public int getLowerBound () {
		return bound;
	}
		
	public IDataBuffer getInputBuffer (int ndx) {
		return inputs[ndx];
	}
	
	public IDataBuffer [] getInputBuffers () {
		return inputs;
	}
	
	public int getTaskSize (int ndx) {
		return tasksize[ndx];
	}
	
	public int getBufferStartPointer (int ndx) {
		return start[ndx];
	}

	public int [] getBufferStartPointers () {
		return start;
	}
	
	public int getBufferEndPointer (int ndx) {
		return end[ndx];
	}

	public int [] getBufferEndPointers () {
		return end;
	}
	
	public long getFreeOffset (int idx) {
		return free[idx];
	}
	
	public long [] getFreeOffsets () {
		return free;
	}
	
	public CrossbowLinkedList<IDataBuffer> getOutput (int ndx) {
		return outputs[ndx];
	}
	
	public CrossbowLinkedList<IDataBuffer> [] getOutputs () {
		return outputs;
	}
	
	public void setOutput (int ndx, IDataBuffer buffer) {
		
		if (! buffer.isFinalised())
			throw new IllegalStateException (String.format("error: output buffer is not finalised (index=%d, batch=%d)", ndx, id));
		
		outputs[ndx].append(buffer);
	}
	
	public boolean isInitialised () {
		return initialised;
	}
	
	public ModelGradient getModelGradient () {
		/*
		if (gradient == null)
			throw new NullPointerException ("error: model gradient is null");
		*/
		return gradient;
	}
	
	public ModelGradient getModelGradient (Model model) {
		
		if (gradient == null)
			gradient = model.getGradientInstance (id);
		return gradient;
	}
	
	public void clear () {
		
		/* Release all outputs */ 
		
		for (int i = 0; i < outputs.length; ++i) {
			
			while (! outputs[i].isEmpty()) {
				IDataBuffer b = outputs[i].removeFirst();
				if (b.referenceCountDecrementAndGet() == 0) {
					b.free();
				}
			}
		}
		
		initialised = false;
	}
	
	public void setLoss (float loss) {
		this.loss = loss;
	}
	
	public float getLoss () {
		return loss;
	}
	
	public void setAccuracy (float accuracy) {
		this.accuracy = accuracy;
	}
	
	public float getAccuracy () {
		return accuracy;
	}
}
