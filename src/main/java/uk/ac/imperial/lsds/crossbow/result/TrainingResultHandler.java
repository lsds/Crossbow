package uk.ac.imperial.lsds.crossbow.result;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.locks.LockSupport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.WorkClock;
import uk.ac.imperial.lsds.crossbow.device.dataset.LightWeightDatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.model.ModelGradient;
import uk.ac.imperial.lsds.crossbow.model.ModelManager;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

@SuppressWarnings("restriction")
public class TrainingResultHandler extends ResultHandler {
	
	private final static Logger log = LogManager.getLogger (TrainingResultHandler.class);
	
	int slots, bytes;
	
	int wpc;
	WorkClock clock;
	
	/*
	 * `f` and `N` ought to be thread-safe, because they are modified
	 * by a single thread (the ResultCollector).
	 */
	float accumulatedLoss;     /* Accumulated loss value */
	float accumulatedAccuracy; /* Accumulated accuracy value */
	int N;   /* Number of accumulated values */
	
	boolean first;
	
	MeasurementQueue individualMeasurements;
	
	public TrainingResultHandler (Dataflow df) {
		
		dataflow = df;
		
		slots = SystemConf.getInstance().numberOfResultSlots();
		wpc = ModelConf.getInstance().getWpc();
		
		clock = new WorkClock (-1, wpc, slots);
		/* Set clock to 0 */
		clock.incrementAndGetNext(null);
		
		/* Reset capacity */
		slots = clock.getMax();
		bytes = getOffset(slots);
		
		log.debug(String.format("%d slots (%d bytes) in training result handler", slots, bytes));
		
		freeBuffers = null;
		
		results = ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN);
		base = theUnsafe.getLong(results, addressOffset);

		gradients = new SlottedObjectPool<ModelGradient> (slots); 

		for (int i = 0; i < slots; i++) {
			
			/* We need a way to differentiate between CPU and GPU tasks.
			 * 
			 * If a slot is set but the gradient is null, then it must
			 * have been set by the GPU.
			 */
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
		
		accumulatedLoss = 0;
		accumulatedAccuracy = 0;
		N = 0;
		
		/* Initialise an append-only queue to store measurements, pooling 1000 nodes to begin with */
		if (SystemConf.getInstance().queueMeasurements())
			queue = new MeasurementQueue(dataflow.getPhase(), 1000, false);
		
		individualMeasurements = new MeasurementQueue(dataflow.getPhase(), ModelConf.getInstance().numberOfTasksPerEpoch(), true);
		
		/* We always store the loss of the very first task */
		first = false;
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
	
	/*
	 * This is thread-safe: it is called by the ResultCollector
	 */
	public void freeSlot (int next) {
		
		/* Preprocess function also accumulates the gradients */
		preprocess (next);

		/* Try synchronise models */
		ModelManager modelManager = dataflow.getExecutionContext().getModelManager();
		if (clock.isBarrier(next)) {
			
			int currentClock = clock.getClock();
			
			if (log.isDebugEnabled())
				log.info(String.format("Synchronise models at clock %4d", currentClock));
			
			/*
			int [] steps = ModelConf.getInstance().getSolverConf().getStepValues();
			boolean found = false;
			for (int i = 0; i < steps.length; ++i) {
				if (currentClock * clock.wpc == steps[i]) {
					found = true;
					break;
				}
			}
			if (found) {
				log.info("Changing clock...");
				clock.wpc = clock.wpc / 2;
				//clock.incrementAndGetNext(null);
			}
			*/
			
			if (! modelManager.trySynchronise(currentClock)) {
				log.warn(String.format("Failed to synchronise models at clock %d (batch %d)", currentClock, next));
			}
		}
		
		ModelGradient gradient = gradients.elementAt(next);
		gradients.setElementAt(next, null);
		if (gradient != null) {
			gradient.free();
			// FIXME
//			if (gradients[next].getParent().requireLastGradient() == false) { // If momentum is not used
//				gradients[next].setLastGradientStatus(false); // Force last-status to be false
//			}
//			gradients[next].setAccumulatedStatus(true);
//
//			gradients[next].tryFree();
//			gradients[next] = null;
		}
		
		postprocess ();
	}
	
	public int getNext () {
		int result = clock.getNext();
		return result;
	}
	
	@Override
	protected void postprocess () {
		clock.incrementAndGetNext(null);
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
					LightWeightDatasetMemoryManager.getInstance().release(Phase.TRAIN.getId(), offset);
			}
			
			log.debug(String.format("Free offset for %6d.%d is %d", next, i, offset));
			freeBuffers[i].free(offset);
			idx += 8;
		}

		/* Accumulate gradients in model manager
		 * 
		 * If null, then it must have been a GPU task.
		 */
		ModelGradient gradient = gradients.elementAt(next);
		if (gradient != null)
			dataflow.getExecutionContext().getModelManager().accumulateGradient(gradient);

		if (SystemConf.getInstance().getDisplayInterval() > 0) {
			/* 
			 * The loss is at position `pos + 20`.
			 * 
			 * If we are accumulating loss values, we read values from all slots.
			 * Otherwise, we read only those values that will be printed.
			 * 
			 * We always increment the counter.
			 */
			N += 1;
			
			if (SystemConf.getInstance().displayAccumulatedLossValue()) {
				accumulatedLoss = accumulatedLoss + results.getFloat(pos + 20);
				accumulatedAccuracy = accumulatedAccuracy + results.getFloat(pos + 24);
			}
			
			individualMeasurements.add(System.nanoTime(), results.getFloat(pos + 20), results.getFloat(pos + 24));
			
			/* If the display interval is 1, we will display (or enqueue) the first measurement anyway */
			if (SystemConf.getInstance().getDisplayInterval() > 1) {
				/* Get the first measurement */
				if (! first) {
					if (SystemConf.getInstance().queueMeasurements()) {
						
                        queue.add(System.nanoTime(), results.getFloat(pos + 20), results.getFloat(pos + 24));
						
                        /* Also display measurements on screen */
						if (SystemConf.getInstance().teeMeasurements())
							log.info(String.format("Training loss (slot #%04d) is %5.5f and accuracy is %5.5f", next, results.getFloat(pos + 20), results.getFloat(pos + 24)));
					} else {
						log.info(String.format("Training loss (slot #%04d) is %5.5f and accuracy is %5.5f", next, results.getFloat(pos + 20), results.getFloat(pos + 24)));
					}
					
					first = true;
				}
			}
			
			if (N == SystemConf.getInstance().getDisplayInterval()) {
				
				if (SystemConf.getInstance().displayAccumulatedLossValue()) {
					/* Compute average loss for the last N u-batches */
					accumulatedLoss /= (float) N;
					accumulatedAccuracy /= (float) N;
				} else {
					/* Read the loss value and accuracy for that particular u-batch */
					accumulatedLoss = results.getFloat(pos + 20);
					accumulatedAccuracy = results.getFloat(pos + 24);
				}
				
				if (SystemConf.getInstance().queueMeasurements()) {
					
					queue.add(System.nanoTime(), individualMeasurements.getAccumulatedAverageLoss(), individualMeasurements.getAccumulatedAverageAccuracy());
					
                    /* Also display measurements on screen */
                    if (SystemConf.getInstance().teeMeasurements())
					    log.info(String.format("Training loss (slot #%04d) is %5.5f (%5.5f) and accuracy is %5.5f (%5.5f)", 
							next, 
							individualMeasurements.getAccumulatedAverageLoss(),
							results.getFloat(pos + 20),
							individualMeasurements.getAccumulatedAverageAccuracy(),
							results.getFloat(pos + 24)
							));

				} else {
					
					/* log.info(String.format("Training loss (slot #%04d) is %5.5f and accuracy is %5.5f", next, accumulatedLoss, accumulatedAccuracy)); */
					
					log.info(String.format("Training loss (slot #%04d) is %5.5f (%5.5f) and accuracy is %5.5f (%5.5f)",
							next, 
							individualMeasurements.getAccumulatedAverageLoss(), 
							results.getFloat(pos + 20),
							individualMeasurements.getAccumulatedAverageAccuracy(),
							results.getFloat(pos + 24)
							));
				}
				
				if (target > individualMeasurements.getAccumulatedAverageLoss()) {
					
					if (finish != null)
						finish.set(true);
				}
				
				/* Reset */
				accumulatedLoss = 0;
				accumulatedAccuracy = 0;
				N = 0;
			}
		}
		
		/* Release the current slot; other threads may use it */
		if (! theUnsafe.compareAndSwapInt(null, getAddress(next), 3, 0))
			throw new IllegalStateException (String.format("error: failed to set slot %d (@%d) to 0", next, pos));
	}
	
	public int numberOfSlots () {
		return slots;
	}
	
	public void flush () {
		
		if (! SystemConf.getInstance().queueMeasurements())
			return;
		
		if (! SystemConf.getInstance().displayAccumulatedLossValue())
			return;
		
		/* Flush last measurement, if N > 0 */
		if (N > 0) {
			log.info("Flushing left-over training task measurements...");
			queue.add(System.nanoTime(), accumulatedLoss / (float) N, accumulatedAccuracy / (float) N);
		}
		
		return;
	}
}
