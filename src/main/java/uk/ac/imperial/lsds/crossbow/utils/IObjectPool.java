package uk.ac.imperial.lsds.crossbow.utils;

public interface IObjectPool<T> {
	
	public T getInstance();
	
	public void free (T item);
}
