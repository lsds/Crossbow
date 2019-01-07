package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import uk.ac.imperial.lsds.crossbow.utils.Queued;

public interface ITaskQueue<T extends Queued> {
	
	public int size ();
	
	public boolean add (T item);
	
	public T poll ();
}
