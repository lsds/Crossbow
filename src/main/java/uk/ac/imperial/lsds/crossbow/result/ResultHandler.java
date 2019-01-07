package uk.ac.imperial.lsds.crossbow.result;

import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

import sun.misc.Unsafe;
import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;
import uk.ac.imperial.lsds.crossbow.dispatcher.ITaskDispatcher;
import uk.ac.imperial.lsds.crossbow.model.ModelGradient;
import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

@SuppressWarnings("restriction")
public abstract class ResultHandler implements IResultHandler {
	
	protected Dataflow dataflow = null;
	
	protected VirtualCircularDataBuffer [] freeBuffers;
	
	protected ITaskDispatcher dispatcher;
	
	protected final static int SLOT_OFFSET = 64;
	
	protected MeasurementQueue queue = null;
	
	/* Set finish to true when `target` loss is reached */
	protected float target = 0;
	protected AtomicBoolean finish = null;
	
	protected static Unsafe   theUnsafe;
	protected static long addressOffset;
	
	static {	
		try {
			
			Field unsafe = Unsafe.class.getDeclaredField("theUnsafe");
			unsafe.setAccessible(true);
			theUnsafe = (Unsafe) unsafe.get (null);
			
			addressOffset = theUnsafe.objectFieldOffset(Buffer.class.getDeclaredField("address"));
			
		} catch (Exception e) {
			throw new AssertionError(e);
		}
	}
	
	/* 
	 * flag: int (4)
	 * offset[0]: long (8)
	 * offset[1]: long (8)
	 * loss: float (4)
	 * count/accuracy: int (4)
	 * 
	 * Flag values:
	 * 
	 *   0: slot is free
	 *   1: slot is being populated by a thread
	 *   2: slot is occupied, but "unlocked"
	 *   3: slot is occupied, but "locked" (somebody is working on it)
	 */
	protected ByteBuffer results;
	protected long base;
	
	protected SlottedObjectPool<ModelGradient> gradients;
	
	public IResultHandler setup () {
		freeBuffers = dataflow.getTaskDispatcher().getBuffers();
		dispatcher  = dataflow.getTaskDispatcher();
		return this;
	}
	
	public int getOffset (int idx) {
		return (idx * SLOT_OFFSET);
	}
	
	public long getAddress (int idx) {
		return base + (long) getOffset(idx);
	}
	
	public boolean ready (int next) {
		
		return (theUnsafe.compareAndSwapInt(null, getAddress(next), 2, 3));
	}
	
	public ByteBuffer getResultSlots () {
		return results;
	}
	
	protected abstract void postprocess ();
	
	protected void preprocess (int next) {
		
		/* Free input buffer */
		int pos = getOffset (next);
		int idx = pos + 4;
		for (int i = 0; i < 2; ++i) {
			long offset = results.getLong(idx);
			/* System.out.println(String.format("[DBG] free offset for %6d.%d is %d", next, i, offset)); */
			freeBuffers[i].free(offset);
			idx += 8;
		}
		
		/* Release the current slot; other threads may use it */
		if (! theUnsafe.compareAndSwapInt(null, getAddress(next), 3, 0))
			throw new IllegalStateException (String.format("error: failed to set slot %d (@%d) to 0", next, pos));
	}
	
	public MeasurementQueue getMeasurementQueue() {
		return queue;
	}
	
	public void setTargetLoss (float target, AtomicBoolean finish) {
		
		if (finish.get())
			throw new IllegalStateException ("error: result handler flag must be false");
		
		this.target = target;
		this.finish = finish;
	}
}
