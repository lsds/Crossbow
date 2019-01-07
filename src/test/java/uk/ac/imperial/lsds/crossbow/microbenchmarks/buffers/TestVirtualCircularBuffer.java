package uk.ac.imperial.lsds.crossbow.microbenchmarks.buffers;

import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;

public class TestVirtualCircularBuffer {
	
	public static void main (String [] args) {
		
		VirtualCircularDataBuffer circularBuffer = new VirtualCircularDataBuffer (null, 1048576, 4096);
		
		/* ... */
		long ret;
		
		int count = 0;
		
		while ((ret = circularBuffer.shift (1024)) >= 0) {
			circularBuffer.debug();
			count ++;
		}
		
		System.out.println("ret = " + ret + " count = " + count);
		
		circularBuffer.free(1023);
		
		while ((ret = circularBuffer.shift (1024)) >= 0) {
			circularBuffer.debug();
			count ++;
		}
		
		System.out.println("ret = " + ret + " count = " + count);
		
		System.out.println ("Bye.");
	}
}
