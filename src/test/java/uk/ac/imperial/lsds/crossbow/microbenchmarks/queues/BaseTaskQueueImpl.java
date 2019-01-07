package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import java.util.concurrent.ConcurrentLinkedQueue;

import uk.ac.imperial.lsds.crossbow.utils.Queued;

public class BaseTaskQueueImpl<T extends Queued> implements ITaskQueue<T> {
	
	private ConcurrentLinkedQueue<T> queue;
	
	public BaseTaskQueueImpl () {
		queue = new ConcurrentLinkedQueue<T>();
	}
	
	public int size () {
		return queue.size();
	}
	
	public boolean add (T item) {
		return queue.offer(item);
	}

	public T poll () {
		return queue.poll();
	}
}
