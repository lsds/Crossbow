package uk.ac.imperial.lsds.crossbow.utils;

public interface IObjectFactory<T> {
	
	public T newInstance ();
	
	public T newInstance (int ndx);
}
