package uk.ac.imperial.lsds.crossbow.result;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.locks.LockSupport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.device.dataset.LightWeightDatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.model.ModelGradient;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

@SuppressWarnings("restriction")
public class TestResultHandler extends ResultHandler {
	
	private final static Logger log = LogManager.getLogger (TestResultHandler.class);
	
	private int slots, bytes;
	
	private int next;
	
	float accumulatedLoss, accumulatedAccuracy; /* Accumulated loss and accuracy values */
	int N; /* Number of accumulated values */
	
	int interval;
	
	public TestResultHandler (Dataflow df) {
		
		dataflow = df;
		
		slots = ModelConf.getInstance().numberOfTestTasks ();
		bytes = getOffset(slots);
		
		log.debug(String.format("%d slots (%d bytes) in test result handler", slots, bytes));
		
		freeBuffers = null;
		
		results = ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN);
		base = theUnsafe.getLong(results, addressOffset);
		
		gradients = new SlottedObjectPool<ModelGradient> (slots); 
		
		for (int i = 0; i < slots; i++) {
			
			gradients.setElementAt (i, null);
			
			int pos = getOffset(i);
			results.position(pos);
			
			results.putInt (0);
			
			for (int j = 0; j < 2; j++)
				results.putLong (Long.MIN_VALUE);
			
			results.putFloat(0); /* Loss */
			results.putFloat(0); /* Accuracy */
			
			/* 28 out of SLOT_OFFSET bytes written. Fill in the rest. */
			for (int j = 0; j < (SLOT_OFFSET - 28); j++)
				results.put((byte) 0);
		}
		
		next = 0;
		
		accumulatedLoss = 0;
		accumulatedAccuracy = 0;
		N = 0;
		
		interval = 0;
		
		/* Initialise an append-only queue to store measurements, pooling 1000 nodes to begin with */
		if (SystemConf.getInstance().queueMeasurements())
			queue = new MeasurementQueue(dataflow.getPhase(), 100, false);
	}
	
	public void setSlot (int taskid, long [] free, float loss, float accuracy, ModelGradient gradient, boolean GPU) {
		
		if (taskid < 0) /* Invalid task id */
			return ;
		
		int idx = ((taskid - 1) % slots);
		
		int offset = getOffset (idx);
		
		while (! theUnsafe.compareAndSwapInt(null, getAddress(idx), 0, 1)) {
			
			if (log.isWarnEnabled())
				log.warn(String.format("warning: task processor (%s) blocked at task %4d (index %d)", Thread.currentThread(), taskid, idx));
			
			LockSupport.parkNanos(1L);
		}
		
		results.putLong(offset +  4, free[0]);
		results.putLong(offset + 12, free[1]);
		
		/* Set loss */
		results.putFloat(offset + 20, loss);
		/* Set accuracy */
		results.putFloat(offset + 24, accuracy);
		/* Set gradient */
		gradients.setElementAt(idx, gradient);
		
		/* No other thread can modify this slot */
		if (! theUnsafe.compareAndSwapInt(null, getAddress(idx), 1, 2))
			throw new IllegalStateException (String.format("error: failed to set slot %d (@%d) to 2", idx, offset));
	}
	
	public void freeSlot (int next) {
		
		preprocess (next);
		
		if (gradients.elementAt(next) != null) {
			throw new IllegalStateException (String.format("gradient must be null"));
		}
		
		postprocess ();
	}
	
	public int getNext () {
		return next;
	}
	
	@Override
	protected void postprocess () {
		next ++;
		if (next == numberOfSlots())
			next = 0;
	}
	
	@Override
	protected void preprocess (int next) {
		/* Free input buffer */
		int pos = getOffset (next);
		int idx = pos + 4;
		for (int i = 0; i < 2; ++i) {
			
			long offset = results.getLong(idx);
			
			if (ModelConf.getInstance().isLightWeight()) {
				/* Release dataset slot (once) */
				if (i == 0)
					LightWeightDatasetMemoryManager.getInstance().release(Phase.CHECK.getId(), offset);
			}
			
			log.debug(String.format("Free offset for %6d.%d is %d", next, i, offset));
			freeBuffers[i].free(offset);
			idx += 8;
		}
		
		/* 
		 * The loss is at position `pos + 12` and accuracy at `pos + 16`.
		 * 
		 * In the testing phase, we accumulate all values.
		 */
		N += 1;
		
		if (N == 1) {
			log.info(String.format("First test task finished at %d", System.nanoTime()));
		}
		
		accumulatedLoss = accumulatedLoss + results.getFloat(pos + 20);
		accumulatedAccuracy = accumulatedAccuracy + results.getFloat(pos + 24);
		
		if (N == numberOfSlots()) {
			
			log.info(String.format("Test finished at %d", System.nanoTime()));
			
			/* Compute average loss for the last N u-batches */
			accumulatedLoss /= (float) N;
			accumulatedAccuracy /= (float) N;
			
			if (SystemConf.getInstance().queueMeasurements ()) {
				queue.add (System.nanoTime(), accumulatedLoss, accumulatedAccuracy);
				/* 
				 * Check if at least N (say, 10) test accuracy values
				 * are below a certain threshold (say, 0.1).
				 * 
				 * If yes, stop execution abruptly. 
				 * 
				 * queue.check (0.1, 10); 
				 */
				/* Also display measurements on screen */
				if (SystemConf.getInstance().teeMeasurements())
					log.info(String.format("[%03d] Test accuracy is %5.5f (loss is %5.5f)", ++interval, accumulatedAccuracy, accumulatedLoss));
			} else {
				log.info(String.format("[%03d] Test accuracy is %5.5f (loss is %5.5f)", ++interval, accumulatedAccuracy, accumulatedLoss));
			}
			
			/* Reset */
			accumulatedLoss = accumulatedAccuracy = 0;
			N = 0;
		}
		
		
		/* Release the current slot; other threads may use it */
		if (! theUnsafe.compareAndSwapInt(null, getAddress(next), 3, 0))
			throw new IllegalStateException (String.format("error: failed to set slot %d (@%d) to 0", next, pos));
	}

	public int numberOfSlots () {
		return slots;
	}
	
	public void flush () {
		return;
	}
}
