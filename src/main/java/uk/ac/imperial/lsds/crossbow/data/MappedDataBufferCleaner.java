package uk.ac.imperial.lsds.crossbow.data;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.MappedByteBuffer;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;

public class MappedDataBufferCleaner {

	private static final boolean SUPPORTED;

	static {
		
		boolean v;
		
		try {
			
			Class.forName("sun.misc.Cleaner");
			Class.forName("java.nio.DirectByteBuffer").getMethod("cleaner");
			
			v = true;
			
		} catch (Exception e) {
			
			v = false;
		}
		
		SUPPORTED = v;
	}
	
	public static void unmap (final MappedByteBuffer buffer) {
		
		if (! SUPPORTED)
			return;
		
		try {
			
			AccessController.doPrivileged (new PrivilegedExceptionAction<Object> () {
				
				public Object run () throws Exception {
					
					final Method cleanerMethod = buffer.getClass ().getMethod ("cleaner");
					cleanerMethod.setAccessible (true);
					
					final Object cleaner = cleanerMethod.invoke (buffer);
					
					if (cleaner != null) {
						
						cleaner.getClass ().getMethod ("clean").invoke (cleaner);
					}
					
					return null;
				}
			}
			);
		
		} catch (PrivilegedActionException e) {
			
			final IOException exception = new IOException ("error: failed to unmap buffer");
			exception.initCause (e.getCause ());
			e.printStackTrace (System.err);
		}
	}
}

