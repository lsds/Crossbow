package uk.ac.imperial.lsds.crossbow.preprocess;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class DataTuple {
	
	private ByteBuffer buffer = null;
	
	private Shape shape;
	
	private DataType type;
	
	public DataTuple () {
		
		this (null, null);
	}
	
	public DataTuple (Shape shape, DataType type) {
		
		this.shape = shape;
		this.type = type;
	}
	
	public DataTuple setShape (Shape shape) {
		
		this.shape = shape;
		
		return this;
	}
	
	public Shape getShape () {
		
		return shape;
	}
	
	public DataTuple setDataType (DataType type) {
		
		this.type = type;
		
		return this;
	}
	
	public DataType getDataType () {
		
		return type;
	}
	
	public ByteBuffer getBuffer () {
		
		if (buffer == null) 
			buffer = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		
		return buffer;
	}
	
	public boolean isShaped () {
		
		return (shape != null && type != null);
	}
	
	public int size () {
		
		return (shape.countAllElements() * type.sizeOf());
	}
}
