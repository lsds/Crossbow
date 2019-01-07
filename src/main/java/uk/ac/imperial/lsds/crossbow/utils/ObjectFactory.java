package uk.ac.imperial.lsds.crossbow.utils;

public interface ObjectFactory<T> {
	
	public T newInstance();
	
	public T newInstance(int ndx);
}
