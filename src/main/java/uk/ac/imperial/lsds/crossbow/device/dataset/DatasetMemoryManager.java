package uk.ac.imperial.lsds.crossbow.device.dataset;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.types.HandlerType;

public class DatasetMemoryManager {

	private static final DatasetMemoryManager instance = new DatasetMemoryManager ();
	
	public static DatasetMemoryManager getInstance () { return instance; }
	
	private boolean loaded;
	
	public DatasetMemoryManager () {
		loaded = false;
	}
	
	public boolean isLoaded () {
		return loaded;
	}
	
	public void init () {
		if (! loaded) {
			String library = String.format("%s/clib-multigpu/libdataset.so", SystemConf.getInstance().getHomeDirectory());
			try {
				System.load (library);
			} catch (final UnsatisfiedLinkError e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			loaded = true;
		}
		init (SystemConf.getInstance().numberOfFileHandlers(), SystemConf.getInstance().getCoreMapper().getOffset(HandlerType.DATASET));
	}
	
	public native int init (int handlers, int offset);
	
	public native int init (int phase, int parts, boolean gpu, int [] block);
	
	public native int configure (int phase, int []  padding);
	public native int configure (int phase, boolean [] copy);
	
	public native int finalise (int phase);
	
	public native int free ();
	
	public native int register (int phase, int type, int id, String filename);
	public native long address (int phase, int type, int id);
	public native int capacity (int phase, int type, int id);
	
	public native int slideOut (int phase, int id);
	public native int slideIn  (int pahse, int id);
}
