package uk.ac.imperial.lsds.crossbow.device;

import uk.ac.imperial.lsds.crossbow.SystemConf;

public class TheCPU {
	
	private static final TheCPU cpuInstance = new TheCPU ();
	
	public static TheCPU getInstance () { return cpuInstance; }
	
	private boolean loaded;
	
	public TheCPU () {
		loaded = false;
	}
	
	public boolean isLoaded () {
		return loaded;
	}
	
	public void init () {
		if (! loaded) {
			String library = String.format("%s/clib-multigpu/libCPU.so", SystemConf.getInstance().getHomeDirectory());
			try {
				System.load (library);
			} catch (final UnsatisfiedLinkError e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			loaded = true;
		}
	}
	
	/* Thread affinity functions */
	
	public native int getNumCores ();
	public native int bind (int cpu);
	public native int unbind ();
	public native int getCpuId ();
}
