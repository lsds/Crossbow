package uk.ac.imperial.lsds.crossbow.model;

import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.PerformanceMonitor;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;

public class ModelManager {

	private final static Logger log = LogManager.getLogger (ModelManager.class);

	private Model [] replicas;
	
	private Model theModel;
	
	private ConcurrentLinkedQueue<Integer> pool;
	
	private ModelGradient accumulatedGradient = null;
	private boolean clear;
	
	/* Count number of locks acquired during "lock-down" */
	private int count;
	private boolean [] locked;
	
	private int checkpointStep;
	
	private PerformanceMonitor monitor;
	
	private boolean autotuning;
	
	private double throughput;
	
	private int step;
	
	public ModelManager (Model theModel) {
		
		this.theModel = theModel;
		replicas = new Model [SystemConf.getInstance().numberOfCPUModelReplicas()];
		
		for (int i = 0; i < replicas.length; ++i) {
			replicas[i] = this.theModel.copy();
			replicas[i].setBaseModel (theModel);
		}
		
		/* The pool contains indices to model replicas:
		 * 
		 *  1, 2, ..., N, 1, 2, ..., N, ...; repeat R times, 
		 *  for N models and R readers/model
		 */
		
		pool = new ConcurrentLinkedQueue<Integer>();
		for (int i = 0; i < SystemConf.getInstance().numberOfReadersPerModel(); ++i) {
			for (int j = 0; j < replicas.length; ++j) {
				pool.offer(new Integer(j));
			}
		}
		
		accumulatedGradient = theModel.getGradientInstance (Integer.MAX_VALUE);
		clear = true;
		
		count = 0;
		locked = new boolean [replicas.length];
		
		checkpointStep = SystemConf.getInstance().getCheckpointInterval ();
		if (checkpointStep % ModelConf.getInstance().getWpc() != 0) {
			while (checkpointStep % ModelConf.getInstance().getWpc() != 0) {
				checkpointStep++;
			}
		}
		checkpointStep /= ModelConf.getInstance().getWpc();
		log.info(String.format("Checkpoint model(s) every %d cycles", checkpointStep));
		
		monitor = null;
		
		if (SystemConf.getInstance().autotuneModels())
			autotuning = true;
		
		throughput = 0D;
		step = 0;
	}
	
	public ModelManager setPerformanceMonitor (PerformanceMonitor monitor) {
		this.monitor = monitor;
		return this;
	}
	
	/**
	 * Accumulate the gradient computed for one of the replicas with the 
	 * gradient of the CPU base model
	 */
	public void accumulateGradient (ModelGradient computedGradient) {
		
		log.debug ("Accumulate gradient from batch " + computedGradient.getMicroBatchId());
		
		IDataBuffer X, Y;
		
		ModelIterator<VariableGradient> g =    computedGradient.iterator();
		ModelIterator<VariableGradient> m = accumulatedGradient.iterator();
		int count;
		
		float beta;
		if (clear) {
			beta = 0F;
			clear = false;
		} else {
			beta = 1F;
		}
		
		while (m.hasNext() && g.hasNext()) {
			
			Y = m.next().getDataBuffer();
			X = g.next().getDataBuffer();
			
			count = Y.limit() / Y.getType().sizeOf();
			
			BLAS.getInstance().saxpby(count, 1, X, 0, X.limit(), /* incX */ 1, beta, Y, /* incY */ 1);
		}
	}
	
	public Integer acquireAccess (int [] clock) {
		
		Integer replicaId;
		Model m;
		
		if ((replicaId = pool.poll()) != null) {
			m = replicas[replicaId.intValue()];
			clock[0] = m.getModelClock();
		}
		return replicaId;
	}

	public Integer upgradeAccess (Integer replicaId, int [] clock) {
		if (replicaId == null) {
			return acquireAccess(clock);
		} else {
			clock[0] = replicas[replicaId.intValue()].getModelClock();
			return replicaId;
		}
	}

	public void release (Integer replicaId) {
		if (replicaId == null)
			return;
		pool.offer(replicaId);
	}

	public Model getModel (Integer replicaId) {
		int ndx = replicaId.intValue();
		return replicas[ndx];
	}
	
	/* Iterate over all models and (un)lock them */
	
	@SuppressWarnings("unused")
	private boolean lockAll () {
		Arrays.fill(locked, false);
		count = 0;
		for (int i = 0; i < replicas.length; ++i) {
			if (replicas[i].tryWriteLock()) {
				++count;
				locked[i] = true;
			}
		}
		return (count == replicas.length);
	}
	
	@SuppressWarnings("unused")
	private int lockAny () {
		Arrays.fill(locked, false);
		count = 0;
		for (int i = 0; i < replicas.length; ++i) {
			if (replicas[i].tryWriteLock()) {
				++count;
				locked[i] = true;
			}
		}
		return count;
	}
	
	public boolean unlockAny () {
		for (int i = 0; i < replicas.length; ++i) {
			if (locked[i]) {
				replicas[i].writeUnlock();
			}
		}
		return true;
	}

	@SuppressWarnings("unused")
	private int merge (int clock) {
		
		for (int i = 0; i < replicas.length; ++i) {
			
			if (locked[i]) {
				
				/* Copy CPU base model to model replica */
				ModelIterator<Variable> m =  theModel.iterator();
				ModelIterator<Variable> r = replicas[i].iterator();
				
				while (m.hasNext() && r.hasNext()) {
					
					IDataBuffer Y = m.next().getDataBuffer();
					IDataBuffer X = r.next().getDataBuffer();
					
					IDataBufferIterator b = Y.getIterator();
					while (b.hasNext()) {
						int offset = b.next();
						X.putFloat(offset, Y.getFloat(offset));
					}
				}
				
				replicas[i].setModelClock (clock);
				replicas[i].resetUpdates ();
			}
		}
		return 0;
	}
	
	/*
	 * Apply accumulated CPU gradient to GPU model, and vice versa 
	 * 
	 */
	@SuppressWarnings("unused")
	private void mergeAcrossDevices () {
		
		throw new IllegalStateException("error: cross-device model synchronisation is not supported yet");
	}
	
	private boolean hasThroughputImproved () {
		
		double current, delta;
		
		if (monitor == null)
			throw new NullPointerException ("error: performance monitor is null");
		
		current = monitor.getCurrentThroughput (0);
		delta = (throughput == 0) ? 1D : ((current - throughput) / throughput);
		throughput = current;
		return (delta > SystemConf.getInstance ().getAutotuneThreshold ());
		
		/*
		boolean value = test;
		test = (! test);
		return value;
		*/
	}
	
	private int autotune () {
		
		/* Auto-tune GPU model replicas (add or remove one replica per GPU) */
		if (SystemConf.getInstance().autotuneModels() && autotuning && (((++ step) % SystemConf.getInstance().getAutotuneInterval()) == 0)) {
			
			if (hasThroughputImproved ()) {
				log.info("Add a new model replica per GPU");
				/* TheGPU.getInstance().addModel (); */
				return  1;
			} else {
				log.info("Remove a model replica per GPU");
				/* TheGPU.getInstance().delModel (); */
				autotuning = false;
				return -1;
			}
		}
		return 0;
	}
	
	@SuppressWarnings("unused")
	private void checkpoint (int clock) { 
		if (checkpointStep > 0) {
			if ((clock % checkpointStep) == 0) {
				log.info("Checkpoint model(s) at clock " + clock);
				if (SystemConf.getInstance().getGPU()) {
					TheGPU.getInstance().checkpointModel (SystemConf.getInstance().getCheckpointDirectory());
				}
			}
		}
	}
	
	/*
	 * NOTE to self:
	 * 
	 * Function has been modified for GPU-only execution; model check-pointing is disabled.
	 */
	public boolean trySynchronise (int clock) {
		
		/* Apply accumulated gradient to the CPU base model and synchronise CPU model replicas */
		/*
		if (SystemConf.getInstance().getCPU()) {
			
			this.theModel.apply(accumulatedGradient);
			
			if (SystemConf.getInstance().getSynchronisationModel() == SynchronisationModel.BSP) {
				if (! lockAll())
					throw new IllegalStateException("error: failed to lock all CPU model replicas at synchronisation barrier");
			} else {
				lockAny();
			}
			merge (clock);
		}
		*/
		
		/* 
		 * Synchronise GPU models
		 *  
		 * Distinguishing between BSP, SSP, and ASP is done at the GPU engine level.
		 * For example, if the synchronisation model is BSP, the engine will attempt
		 * to lock all GPU model replicas (otherwise an exception will be thrown).
		 */
		
		/*
		if (SystemConf.getInstance().getGPU ()) {
			
			TheGPU.getInstance().lockAny();
			
			TheGPU.getInstance().merge (SystemConf.getInstance().isHybrid());
			TheGPU.getInstance().synchronise(0, clock, autotune(), false);
		}
		*/
		
		TheGPU.getInstance().lockAny();
		TheGPU.getInstance().synchronise(0, clock, autotune(), false);
		
		/* Synchronise models across CPU and GPU boundary */
		/* if (SystemConf.getInstance().isHybrid()) mergeAcrossDevices (); */
		
		/* Checkpoint model (and other necessary variables) */
		/* checkpoint (clock); */
		
		/* Next time `accumulateGradient` is called clear `accumulatedGradient` */
		clear = true;
		
		/* Unlock models */
		/*
		if (SystemConf.getInstance().getGPU())
			TheGPU.getInstance().unlockAny();
		
		if (SystemConf.getInstance().getCPU()) {
			unlockAny();
		}
		*/
		TheGPU.getInstance().unlockAny();
		
		return true;
	}
	
	public void GPURegister () {
		TheGPU.getInstance().setModelManager
			(SystemConf.getInstance().numberOfGPUModelReplicas(), SystemConf.getInstance().getSynchronisationModel().getId());
	}

	public void dump () {
		for (int i = 0; i < replicas.length; ++i) {
			System.out.print(replicas[i] + " ");
		}
		System.out.println();
	}
	
	/*
	 * This method is called after all tasks have completed.
	 * So all models, both on CPU and GPU, are unlocked.
	 * 
	 * The idea is to synchronise them one last time.
	 */
	public void synchroniseAll() {
		return;
	}

}
