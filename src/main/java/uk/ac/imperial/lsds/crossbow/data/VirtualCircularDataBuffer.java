package uk.ac.imperial.lsds.crossbow.data;

import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.DatasetAddressTranslator;
import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

public class VirtualCircularDataBuffer {
	
	private final static Logger log = LogManager.getLogger (VirtualCircularDataBuffer.class);
	
	private SlottedObjectPool<MappedDataBuffer> buffers;
	
	private long capacity, limit;
	
	private DatasetAddressTranslator translator;
	
	private final PaddedAtomicLong start;
	private final PaddedAtomicLong end;
	
	private AtomicLong bytesProcessed;
	private AtomicLong tasksProcessed;
	
	private PaddedLong h;
	
	private long wraps;
	
	public VirtualCircularDataBuffer (SlottedObjectPool<MappedDataBuffer> buffers, long capacity) {
		
		this(buffers, capacity, capacity);
	}
	
	public VirtualCircularDataBuffer (SlottedObjectPool<MappedDataBuffer> buffers, long capacity, long limit) {
		
		this.buffers = buffers;
		
		this.capacity = capacity;
		this.limit = limit;
		
		if (this.buffers != null)
			translator = new DatasetAddressTranslator (buffers);
		else
			translator = null;
		
		start = new PaddedAtomicLong (0L);
		end = new PaddedAtomicLong (0L);
		
		bytesProcessed = new AtomicLong (0L);
		tasksProcessed = new AtomicLong (0L);
		
		h = new PaddedLong (0L);
		
		wraps = 0;
	}
	
	public long normalise (long index) {
		
		return (index % capacity);
	}
	
	public long shift (int bytes) {
		
		/* log.debug(String.format("Shift %d bytes (capacity = %d, limit = %d)", bytes, capacity, limit)); */
		
		if (bytes <= 0)
			throw new IllegalArgumentException (String.format("error: cannot put %d bytes to circular buffer", bytes));
		
		/* Get the end pointer */
		final long _end = end.get();
		
		/* Find remaining bytes until the circular buffer wraps */
		final long wrapPoint = (_end + bytes - 1) - limit;
		
		if (h.value <= wrapPoint) {
			
			h.value = start.get();
			
			if (h.value <= wrapPoint) {
				
				/* debug (); */
				return -1;
			}
		}
		
		long index = normalise (_end);
		
		if (bytes > (capacity - index))
			throw new IllegalStateException ("error: circular buffer overflow");
		
		long p = normalise (_end + bytes);
		
		if (p <= index)
			wraps ++;
		
		end.lazySet(_end + bytes);
		
		return index;
	}
	
	public void debug () {
		
		long head = start.get();
		long tail = end.get();
		
		long remaining = (tail < head) ? (head - tail) : (capacity - (tail - head));
		
		log.debug (
			String.format(
				"start %20d (%20d) end %20d (%20d) %7d wraps %20d bytes remaining", 
				normalise (head), head, normalise (tail), tail, getWraps(), remaining
			)
		);
	}
	
	public void free (long offset) {
		
		final long _start = start.get();
		final long index = normalise (_start);
		
		final long bytes;
		/* Measurements */
		if (offset <= index)
			bytes = capacity - index + offset + 1;
		else
			bytes = offset - index + 1;
		
		/*
		if (log.isDebugEnabled())
			debug ();
		*/
		
		bytesProcessed.addAndGet(bytes);
		tasksProcessed.incrementAndGet();
		
		/* Set new start pointer */
		start.lazySet(_start + bytes);
	}
	
	public long getBytesProcessed () {
		
		return bytesProcessed.get(); 
	}
	
	public long getTasksProcessed () {
		
		return tasksProcessed.get();
	}
	
	public long getWraps () {
		
		return wraps;
	}
	
	public long capacity () {
		
		return capacity;
	}
	
	public DatasetAddressTranslator getAddressTranslator () {
		
		return translator;
	}

	public long getNormalisedStartPointer() {
		
		return normalise (start.get());
	}
}
