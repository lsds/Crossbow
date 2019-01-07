package uk.ac.imperial.lsds.crossbow.device;

import uk.ac.imperial.lsds.crossbow.SystemConf;

public class ObjectRef {
	
	private static final ObjectRef objectRefInstance = new ObjectRef ();
	
	public static ObjectRef getInstance () { return objectRefInstance; }
	
	public ObjectRef () {
	}
	
	public void init () {
		String library = String.format("%s/clib-multigpu/libobjectref.so", SystemConf.getInstance().getHomeDirectory());
		try {
			System.load (library);
		} catch (final UnsatisfiedLinkError e) {
			System.err.println(e.getMessage());
			System.exit(1);
		}
	}
	
	public native void create (int value);
	public native Integer get ();
	public native int test (Integer var);
	public native Integer testAndGet (Integer var);
}
