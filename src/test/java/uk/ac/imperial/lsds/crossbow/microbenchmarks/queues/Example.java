package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import uk.ac.imperial.lsds.crossbow.utils.IObjectPool;
import uk.ac.imperial.lsds.crossbow.utils.Pooled;
import uk.ac.imperial.lsds.crossbow.utils.Queued;

public class Example implements Pooled<Example>, Queued {
	
	int id;
	
	IObjectPool<Example> pool = null;
	
	public Example () {
		this(0);
	}
	
	public Example (int id) {
		this.id = id;
	}
	
	public void setId (int id) {
		this.id = id;
	}
	
	public void setPool (IObjectPool<Example> pool) {
		this.pool = pool;
	}

	public void free () {
		
		if (pool == null)
			throw new IllegalStateException("error: object pool is null");
		
		pool.free(this);
	}

	public int getKey() {
		return id;
	}
}
