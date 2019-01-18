package uk.ac.imperial.lsds.crossbow.data;

import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.utils.IObjectFactory;

public class CustomDataBufferFactory implements IObjectFactory<DataBuffer> {
	
	private int capacity;
	private DataType type;
	
	public CustomDataBufferFactory (int capacity, DataType type) {
		
		this.capacity = capacity;
		this.type = type;
	}
	
	@Override
	public DataBuffer newInstance () {
		
		return new DataBuffer (0, capacity, type);
	}
	
	@Override
	public DataBuffer newInstance (int ndx) {
		
		return new DataBuffer (ndx, capacity, type);
	}
}
