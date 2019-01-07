package uk.ac.imperial.lsds.crossbow.device.blas;

import java.lang.reflect.Field;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import sun.misc.Unsafe;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;

@SuppressWarnings("restriction")
public class HeapMemoryManager {
	
	private final static Logger log = LogManager.getLogger (HeapMemoryManager.class);
	
	private static Unsafe theUnsafe;
	
	private static Unsafe getUnsafeMemory () {
		
		try {
			
			Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
			theUnsafe.setAccessible(true);
			return (Unsafe) theUnsafe.get (null);
			
		} catch (Exception e) {
			throw new AssertionError(e);
		}
	}
	
	private ConcurrentLinkedQueue<Integer> pointers;
	
	private IDataBuffer [] buffers;
	
	private int [] start;
	private int []   end;
	
	public HeapMemoryManager (int capacity) {
		
		theUnsafe = getUnsafeMemory ();
		
		pointers = new ConcurrentLinkedQueue<Integer> ();
		
		buffers = new IDataBuffer[capacity];
		
		start = new int [capacity];
		  end = new int [capacity];
		
		for (int ndx = 0; ndx < capacity; ++ndx) {
			
			pointers.offer(ndx);
			
			buffers[ndx] = null;
			start[ndx] = end[ndx] = 0;
		}
	}
	
	public void inputDataMovementCallback (int ndx, long address, int size) {
		
		int length = end[ndx] - start[ndx];
		
		if (length <= 0)
			throw new ArrayIndexOutOfBoundsException(String.format("error: invalid offset(s) for buffer %d", ndx));
		
		if (length > size)
			throw new ArrayIndexOutOfBoundsException(String.format("error: writing buffer %d out of bounds", ndx));
		
		if (log.isDebugEnabled())
			log.debug (String.format("Write %d bytes from buffer [%d] to @%d (destination buffer size is %d)", 
				length, ndx, address, size));
		
		theUnsafe.copyMemory (
			buffers[ndx].array(), 
			Unsafe.ARRAY_BYTE_BASE_OFFSET + start[ndx], 
			null, 
			address, 
			length
		);
	}
	
	public void outputDataMovementCallback (int ndx, long address, int size) {
		
		int length = end[ndx] - start[ndx];
		
		int capacity = buffers[ndx].limit();
		
		if (length <= 0 || length != capacity)
			throw new ArrayIndexOutOfBoundsException(String.format("error: invalid offset(s) for buffer %d", ndx));
		
		if (log.isDebugEnabled())
			log.debug (String.format("Read %d bytes from @%d to buffer [%d] (destination buffer size is %d)", 
				length, address, ndx, capacity));
		
		theUnsafe.copyMemory(
			null, 
			address, 
			buffers[ndx].array(), 
			Unsafe.ARRAY_BYTE_BASE_OFFSET, 
			length
			);
	}
	
	public Integer setAndGet (IDataBuffer buffer) {
		
		return setAndGet (buffer, 0, buffer.limit());
	}
	
	public Integer setAndGet (IDataBuffer buffer, int from, int to) {
		
		if (! buffer.isFinalised())
			throw new IllegalStateException("error: cannot transfer data from/to an unfinalised buffer");
		
		Integer p = pointers.remove();
		int ndx = p.intValue();
		buffers[ndx] = buffer;
		start [ndx] = from;
		end [ndx] = to;
		return p;
	}
	
	public void free (Integer bufferId) {
		
		int ndx = bufferId.intValue();
		buffers[ndx] = null;
		start[ndx] = end[ndx] = 0;
		pointers.offer(bufferId);
	}
}
