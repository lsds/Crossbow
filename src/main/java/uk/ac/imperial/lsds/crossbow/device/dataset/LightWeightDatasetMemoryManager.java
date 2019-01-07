package uk.ac.imperial.lsds.crossbow.device.dataset;

import java.nio.ByteBuffer;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.types.HandlerType;

public class LightWeightDatasetMemoryManager {

	private static final LightWeightDatasetMemoryManager instance = new LightWeightDatasetMemoryManager ();
	
	public static LightWeightDatasetMemoryManager getInstance () { return instance; }
	
	private boolean loaded;
	
	public LightWeightDatasetMemoryManager () {
		loaded = false;
	}
	
	public boolean isLoaded () {
		return loaded;
	}
	
	public void init () {
		if (! loaded) {
			String library = String.format("%s/clib-multigpu/liblightweightdataset.so", SystemConf.getInstance().getHomeDirectory());
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
	
	public native int setPadding (int phase, int [] padding);
	public native int setHandler (int phase, ByteBuffer buffer, int limit);
	
	public native int configure (int phase, int batchsize, int elements, int fill);
	
	public native int finalise (int phase);
	
	public native int free ();
	
	public native int  register (int phase, int type, int id, String filename);
	public native long  address (int phase, int type);
	public native int  capacity (int phase, int type);
	
	public native int reserve (int phase, long [] free);
	public native int release (int phase, long    free); /* A single pointer should suffice. */
}
