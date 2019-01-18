package uk.ac.imperial.lsds.crossbow.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.utils.IObjectPool;
import uk.ac.imperial.lsds.crossbow.utils.Pooled;

public class DataBuffer implements IDataBuffer, Pooled<DataBuffer> {

	private int id;
	private int capacity;
	private DataType type;
	
	private ByteBuffer buffer;

	private boolean direct;
	
	/* 
	 * Data buffers have fixed size but they are not necessarily always full. 
	 * When a buffer is finalised, it can no longer be modified: the current
	 * buffer position becomes the limit of the buffer.
	 */
	private boolean finalised;
	
	private IDataBufferIterator iterator;
	
	private IObjectPool<DataBuffer> pool;
	
	private int referenceCount;

	public DataBuffer (int capacity, DataType type) {
		this (0, capacity, type);
	}

	public DataBuffer (int id, int capacity, DataType type) {
		
		if (capacity <= 0)
			throw new IllegalArgumentException("error: buffer size must be greater than 0");
		
		this.id = id;
		this.capacity = capacity;
		this.type = type;
		
		direct = SystemConf.getInstance().useDirectBuffers();
		
		if (! direct) {
			buffer = ByteBuffer.allocate (this.capacity).order (ByteOrder.LITTLE_ENDIAN);
		} else {
			buffer = ByteBuffer.allocateDirect (this.capacity).order (ByteOrder.LITTLE_ENDIAN);
		}
		
		finalised = false;
		iterator = new DataIterator (this);
		
		referenceCount = 0;
	}
	
	private static class DataIterator implements IDataBufferIterator {

		int cursor;
		IDataBuffer buffer;
		int step;
		
		public DataIterator (IDataBuffer b) {
			buffer = b;
			cursor = 0;
			step   = b.getType().sizeOf();
		}
		
		public int next () {
			int result = cursor;
			cursor += step;
			return result;
		}

		public boolean hasNext () {
			return (cursor < buffer.limit());
		}

		public IDataBufferIterator reset () {
			return reset (0);
		}

		public IDataBufferIterator reset (int offset) {
			cursor = offset;
			return this;
		}
	}
	
	@Override
	public void setPool (IObjectPool<DataBuffer> pool) {
		this.pool = pool;
	}
	
	@Override
	public void free () {
		if (pool == null)
			throw new IllegalStateException ("error: buffer pool is not set");
		clear ();
		pool.free(this);
	}
	
	@Override
	public int getBufferId () {
		return id;
	}
	
	@Override
	public DataType getType () {
		return type;
	}
	
	private void checkFinalised () {
		if (! finalised)
			throw new IllegalStateException("error: buffer is not finalised");
	}
	
	@Override
	public byte get (int offset) {
		checkFinalised ();
		return buffer.get (offset);
	}
	
	@Override
	public int getInt (int offset) {
		checkFinalised ();
		return buffer.getInt (offset);
	}
	
	@Override
	public float getFloat (int offset) {
		checkFinalised ();
		return buffer.getFloat (offset);
	}
	
	@Override
	public long getLong (int offset) {
		checkFinalised ();
		return buffer.getLong (offset);
	}
	
	@Override
	public int limit () {
		checkFinalised ();
		return buffer.limit ();
	}
	
	@Override
	public int position () {
		checkFinalised ();
		return buffer.position ();
	}

	@Override
	public int capacity () {
		return capacity; /* Or, buffer.capacity () */
	}
	
	@Override
	public void reset () {
		checkFinalised ();
		buffer.position (0);
	}
	
	@Override
	public void clear () {
		checkFinalised ();
		finalised = false;
		buffer.clear ();
	}
	
	public void put (int index, byte  value) {
		checkFinalised ();
		buffer.put (index, value);
	}
	
	@Override
	public void putInt (int index, int value) {
		checkFinalised ();
		buffer.putInt (index, value);
	}

	@Override
	public void putFloat (int index, float value) {
		checkFinalised ();
		buffer.putFloat (index, value);
	}

	@Override
	public void putLong (int index, long value) {
		checkFinalised ();
		buffer.putLong (index, value);
	}

	@Override
	public boolean isDirect () {
		return direct;
	}
	
	@Override
	public void finalise (int offset) {
	
		if (isFinalised ())
			throw new IllegalStateException ("error: buffer already finalised");
		
		buffer.limit(offset);
		finalised = true;
	}
	
	@Override
	public boolean isFinalised () {
		return finalised;
	}
	
	@Override
	public IDataBufferIterator getIterator () {	
		return iterator.reset();
	}
	
	@Override
	public void put (IDataBuffer b) {
		put(b, 0, b.limit(), true);
	}

	@Override
	public void put (IDataBuffer b, int offset, int length, boolean resetPosition) {
		if (resetPosition)
			reset ();
		else /* Just check */
			checkFinalised ();
		for (int i = offset; i < (offset + length); ++i)
			buffer.put(b.get(i));
	}
	
	@Override
	public void bzero () {
		/* Set position to zero (check finalised) */
		reset ();
		while (buffer.hasRemaining())
			buffer.put ((byte) 0);
		return;
	}
	
	@Override
	public void bzero (int offset, int length) {
		checkFinalised ();
		buffer.position (offset);
		int i = 0;
		while (i++ < length)
			buffer.put((byte) 0);
		return;
	}
	
	@Override
	public ByteBuffer getByteBuffer () {
		return buffer;
	}

	@Override
	public byte [] array () {
		return buffer.array();
	}

	@Override
	public float computeChecksum () {
		
		float checksum = 0;
		
		iterator.reset ();
		while (iterator.hasNext ()) {
			
			int offset = iterator.next();
			
			if (type == DataType.FLOAT) checksum += buffer.getFloat(offset);
			else
				checksum += (float) (buffer.getInt(offset));
		}
		
		return checksum;
	}
	
	@Override
	public int referenceCountGet () {
		return referenceCount;
	}

	@Override
	public int referenceCountGetAndIncrement () {
		int result = referenceCount ++;
		return result;
	}

	@Override
	public int referenceCountDecrementAndGet () {
		return (--referenceCount);
	}
}
