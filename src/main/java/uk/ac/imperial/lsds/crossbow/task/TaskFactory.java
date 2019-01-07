package uk.ac.imperial.lsds.crossbow.task;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.result.IResultHandler;

public class TaskFactory {

	public static AtomicLong count = new AtomicLong (0L);
	
	private static ConcurrentLinkedQueue<Task> pool = new ConcurrentLinkedQueue<Task>();
	
	public static Task newInstance 
		(int taskid, SubGraph graph, Batch batch, IResultHandler handler, Integer replica) {
		
		Task task;
		
		task = pool.poll();
		
		if (task == null) {
			
			count.incrementAndGet();
			return new Task (taskid, graph, batch, handler,replica);
		}
		
		task.set (taskid, graph, batch, handler, replica);
		
		return task;
	}
	
	public static void free (Task task) {
		
		pool.offer (task);
	}
}
