package uk.ac.imperial.lsds.crossbow.processor;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.model.ModelManager;
import uk.ac.imperial.lsds.crossbow.task.TaskQueue;
import uk.ac.imperial.lsds.crossbow.types.ExecutionMode;

public class TaskProcessorPool {

	private final static Logger log = LogManager.getLogger (TaskProcessorPool.class);

	private int workers;
	
	private TaskQueue queue;
	private int [][] matrix;
	
	private ModelManager modelManager;
	
	private TaskProcessor [] processor;
	
	public TaskProcessorPool (ExecutionContext context) {
		
		workers = SystemConf.getInstance().numberOfWorkerThreads();
		
		queue = context.getExecutorQueue();
		matrix = context.getThroughputMatrix();
		
		modelManager = context.getModelManager();
		
		processor = new TaskProcessor[workers];
		
		ExecutionMode mode = SystemConf.getInstance().getExecutionMode();
		
		log.info(String.format("%d threads (mode %s)", workers, mode));
		
		switch (mode) {
		case HYBRID:
			/* Assign the first processor to be the GPU worker */
			processor[0] = new TaskProcessor(0, queue, matrix, modelManager, true);
			for (int i = 1; i < workers; i++)
				processor[i] = new TaskProcessor(i, queue, matrix, modelManager, false);
			break;
		case CPU:
			for (int i = 0; i < workers; i++)
				processor[i] = new TaskProcessor(i, queue, matrix, modelManager, false);
			break;
		case GPU:
			
			if (workers > 1)
				throw new IllegalStateException 
					("error: number of workers must be equal to 1 in GPU-only execution mode");
			
			processor[0] = new TaskProcessor(0, queue, matrix, modelManager, true);
			break;
		default:
			throw new IllegalStateException ("error: invalid execution mode");
		}
	}
	
	public TaskQueue start (Executor executor) {
		for (int i = 0; i < workers; i++)
			executor.execute(processor[i]);
		return queue;
	}
	
	public long getProcessedTasks (int pid, int gid) {
		return processor[pid].getProcessedTasks(gid);
	}
	
	public void stop () throws Exception {
		
		if (SystemConf.getInstance().getExecutionMode().equals(ExecutionMode.GPU) && SystemConf.getInstance().useDirectScheduling())
			/* There should be no task processor running */
			return;
		
		CountDownLatch signal = new CountDownLatch (workers);
		for (int i = 0; i < workers; i++)
			processor[i].shutdown(signal);
		signal.await();
		return;
	}
}
