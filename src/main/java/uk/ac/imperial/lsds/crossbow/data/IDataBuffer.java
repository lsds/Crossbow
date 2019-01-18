package uk.ac.imperial.lsds.crossbow.data;

import java.nio.ByteBuffer;

import uk.ac.imperial.lsds.crossbow.types.DataType;

public interface IDataBuffer {
	
	public int getBufferId ();
	
	public DataType getType ();
	
	public byte  get      (int offset);
	public int   getInt   (int offset);
	public float getFloat (int offset);
	public long  getLong  (int offset);
	
	public int limit ();
	
	public int position ();
	public int capacity ();
	
	public void reset ();
	public void clear ();
	
	public void put      (int index, byte  value);
	public void putInt   (int index, int   value);
	public void putFloat (int index, float value);
	public void putLong  (int index, long  value);
	
	public boolean isDirect ();
	
	public void finalise (int index);
	public boolean isFinalised ();
	
	public IDataBufferIterator getIterator ();
	
	public void put (IDataBuffer buffer);
	public void put (IDataBuffer buffer, int offset, int length, boolean resetPosition);
	
	public void bzero ();
	public void bzero (int offset, int length);
	
	public ByteBuffer getByteBuffer ();
	public byte [] array ();
	
	public void free ();
	
	public int referenceCountGet ();
	public int referenceCountGetAndIncrement ();
	public int referenceCountDecrementAndGet ();

	public float computeChecksum ();
}
