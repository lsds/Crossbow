package uk.ac.imperial.lsds.crossbow.processor;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.device.TheCPU;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.model.ModelManager;
import uk.ac.imperial.lsds.crossbow.task.AbstractTask;
import uk.ac.imperial.lsds.crossbow.task.TaskQueue;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class TaskProcessor implements Runnable {
	
	private final static Logger log = LogManager.getLogger (TaskProcessor.class);
	
	private volatile boolean stop = false;
	private CountDownLatch signal = null;
	
	private TaskQueue queue;
	
	private int [][] matrix;
	
	private int pid;
	private boolean GPU;
	
	private ModelManager modelmanager;
	
	private int deviceId = 0; /* Processor class: GPU (0) or CPU (1) */
	
	/* Measurements */
	private AtomicLong [] tasksProcessed;
	
	private int [] clock = new int [1];
	
	private int tid; /* Thread id */
	
	public TaskProcessor (int pid, TaskQueue queue, int [][] matrix, ModelManager modelmanager, boolean GPU) {
		
		this.pid = pid;
		this.queue = queue;
		this.matrix = matrix;
		
		this.modelmanager = modelmanager;
		
		this.GPU = GPU;
		
		if (GPU) deviceId = 0;
		else 
			deviceId = 1;
		
		int N = matrix[0].length;
		this.tasksProcessed = new AtomicLong [N];
		
		for (int i = 0; i < N; i++)
			this.tasksProcessed[i] = new AtomicLong (0L);
	}
	
	public void run() {
		
		AbstractTask task = null;
		
		if (GPU) {
			
			if (SystemConf.getInstance().useDirectScheduling()) {
				
				log.info(String.format("GPU worker exits (tasks are scheduled directly by the dispatcher)"));
				return;
			} 
			
			log.info(String.format("GPU worker is thread %s", Thread.currentThread()));
			TheCPU.getInstance().bind(1);
			
			Thread.currentThread().setName("GPU task processor");
			
		} else {
			TheCPU.getInstance().bind(pid + 1);
		}
		
		tid = ThreadMap.getInstance().register(Thread.currentThread().getId());
		
		retry: while (! stop) {
			try {
				/*  
				 * Acquire model access to a replica. The return value
				 * is null when all model replicas have been reserved
				 * by other workers.
				 */
				clock[0] = -1;
				
				Integer replicaId;
				
				if (GPU) {
					replicaId = TheGPU.getInstance().acquireAccess (clock);
				} else
					replicaId = modelmanager.acquireAccess (clock);
				
				while ((task = queue.poll(matrix, deviceId, replicaId, clock[0])) == null) {
					LockSupport.parkNanos(1L);
					if (stop)
						continue retry;
					/*
					 * If `replicaId` is null, try to acquire access to a model replica again. 
					 * Otherwise, upgrade the current replica's wpc counter since the previous 
					 * one failed (note that model updates can be applied asynchronously).
					 */
					if (GPU) {
						replicaId = TheGPU.getInstance().upgradeAccess(replicaId, clock);
						if (replicaId == null)
							continue retry;
					} else
						replicaId = modelmanager.upgradeAccess(replicaId, clock);
				}
				/*
				 * Early release
				 *
				 * Either the task's replica id is not null, in which case a model replica has
				 * already been reserved for that task, or it is null and the task access type 
				 * is neither R/O nor R/W (otherwise, we would have assigned replicaId to it).
				 */
				if (replicaId != null) {
					if (task.replicaId != null && task.replicaId != replicaId) {
						if (GPU)
							throw new IllegalStateException ("error: cannot release a GPU model replica too early");
						/* Early release */
						modelmanager.release(replicaId);
					} else if (task.replicaId == null && task.access.compareTo(ModelAccess.NA) > 0) {
						/* Bind task to model replica */
						task.replicaId = replicaId;
					}
				}
				if (log.isDebugEnabled())
					log.debug(String.format("Worker %d (thread %d) got task %d with replica %s and lower bound %d", 
							pid, tid, task.taskId, ((task.replicaId != null) ? task.replicaId.toString() : "null"), task.lowerBound));
				
				task.setGPU(GPU);
				if (! task.isValidationTask())
					tasksProcessed[task.graphId].incrementAndGet();
				
				task.run();
				
				task.free();
				
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		}
		log.info (String.format("Worker %2d shuts down", pid));
		if (signal == null)
			throw new NullPointerException ("error: countdown latch is null");
		signal.countDown();
		return;
	}
	
	public long getProcessedTasks (int gid) {
		return tasksProcessed[gid].get();
	}
	
	public void shutdown (CountDownLatch signal) {
		this.signal = signal;
		stop = true;
	}
}
