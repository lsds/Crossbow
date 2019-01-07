package uk.ac.imperial.lsds.crossbow.utils;

public interface Pooled<T> {
	
	public void setPool (IObjectPool<T> pool);
}
