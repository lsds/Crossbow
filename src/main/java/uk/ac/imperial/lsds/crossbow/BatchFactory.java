package uk.ac.imperial.lsds.crossbow;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;

public class BatchFactory {

	public static AtomicLong count = new AtomicLong (0L);

	private static ConcurrentLinkedQueue<Batch> pool = new ConcurrentLinkedQueue<Batch>();

	public static Batch newInstance 
		(int task, int bound, IDataBuffer [] buffer, int [] tasksize, long [] start, long [] end, long [] free, int numberOfOperators) {
		
		Batch batch;
		batch = pool.poll();
		if (batch == null) {
			count.incrementAndGet();
			return new Batch (task, bound, buffer, tasksize, start, end, free, numberOfOperators);
		}
		batch.set (task, bound, buffer, tasksize, start, end, free);
		return batch;
	}
	
	public static void free (Batch batch) {
		if (batch == null)
			return;
		batch.clear();
		pool.offer (batch);
	}
}
