package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

public class DatasetAddressTranslator {
	
	private final static Logger log = LogManager.getLogger (DatasetAddressTranslator.class);
	
	private SlottedObjectPool<MappedDataBuffer> buffers;
	
	private boolean check;
	
	private int parts;
	
	private long [] start, end; /* Pointers */
	
	public DatasetAddressTranslator (SlottedObjectPool<MappedDataBuffer> buffers) {
		
		this (buffers, false);
	}
	
	public DatasetAddressTranslator (SlottedObjectPool<MappedDataBuffer> buffers, boolean check) {
		
		this.buffers = buffers;
		this.check = check;
		
		parts = buffers.elements();
		
		start = new long [parts];
		end   = new long [parts];
		
		/* Initialise pointers */
		
		setPointers (0, 0, buffers.elementAt (0).getSize ());
		
		for (int idx = 1; idx < parts; ++idx) {
			
			setPointersFromPrevious (idx);
		}
	}
	
	public void setPointers (int idx, long p, long q) {
		
		start [idx] = p;
		end   [idx] = q;
	}
	
	public void setPointersFromPrevious (int idx) {
		
		long p = getEndPointer (idx - 1);
		long q = p + buffers.elementAt (idx).getSize ();
		
		setPointers (idx, p, q);
	}
	
	public void setStartPointer (int idx, long value) {
		
		start [idx] = value;
	}
	
	public long getStartPointer (int idx) {
		
		return start [idx];
	}
	
	public void setEndPointer (int idx, long value) {
		
		end [idx] = value;
	}
	
	public long getEndPointer (int idx) {
		
		return end [idx];
	}
	
	public long normalisePointer (int idx, long value) {
		
		return value - getStartPointer (idx);
	}
	
	public long normalisePointer (long value) {
		
		int idx = searchFor (value);
		return normalisePointer (idx, value);
	}
	
	public int searchFor (long value) {
		
		int l = 0;
		int h = parts - 1;
		
		while (l <= h) {
			
			int m = (l + h) / 2;
			
			if (value >= getEndPointer (m))
				l = m + 1;
			else
			if (value < getStartPointer (m))
				h = m - 1;
			else
				return m;
		}
			
		return -1;
	}
	
	public int translate (int ndx, long [] p, long [] q) {
		
		log.debug(String.format("Translate address [%d] [%d, %d)", ndx, p[ndx], q[ndx]));
		
		int pos = searchFor (p[ndx]);
		
		if (check) {
			
			int __pos = searchFor (q[ndx]);
			
			if (! (pos == __pos)) {
				
				System.err.println("error: invalid address translation");
				System.exit (1);
			}
		}
		
		p [ndx] = normalisePointer (pos, p[ndx]);
		q [ndx] = normalisePointer (pos, q[ndx]);
		
		log.debug(String.format("Return position %2d, [%d, %d)", pos, p[ndx], q[ndx]));
		
		return pos;
	}
	
	public int translate (long p) {
		
		return searchFor (p);
	}
	
	public void dump () {
		
		StringBuilder s = new StringBuilder (String.format("=== [Dataset address translator dump: %d file partitions, %d bytes] ===\n", parts, getEndPointer (parts - 1)));
		
		for (int idx = 0; idx < parts; ++idx) {
			
			s.append(String.format("%3d: [%6d, %6d)\n", idx, getStartPointer (idx), getEndPointer (idx)));
		}
		
		s.append ("=== [End of dataset address translator dump] ===\n");
		
		System.out.println (s.toString());
	}
}
