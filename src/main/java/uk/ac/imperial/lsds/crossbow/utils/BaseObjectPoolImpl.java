package uk.ac.imperial.lsds.crossbow.utils;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

public class BaseObjectPoolImpl<T> implements IObjectPool<T> {
	
	private ConcurrentLinkedQueue<T> pool;
	
	private AtomicLong count;
	
	private IObjectFactory<T> factory;
	
	public BaseObjectPoolImpl (IObjectFactory<T> factory) {
		this (0, factory);
	}
	
	public BaseObjectPoolImpl (int initialCapacity, IObjectFactory<T> factory) {
		
		pool = new ConcurrentLinkedQueue<T>();
		
		this.factory = factory;
		
		for (int i = 0; i < initialCapacity; ++i)
			pool.offer(factory.newInstance());
		
		count = new AtomicLong ((long) initialCapacity);
	}
	
	public T getInstance () {
		
		T t = pool.poll();
		
		if (t == null) {
			count.incrementAndGet();
			t = factory.newInstance();
		}
		return t;
	}
	
	public void free (T item) {
		if (item == null)
			return;
		pool.offer (item);
	}
}
