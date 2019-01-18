package uk.ac.imperial.lsds.crossbow.data;

public interface IDataBufferIterator {
	
	public int next ();
	
	public boolean hasNext ();
	
	public IDataBufferIterator reset ();
}
