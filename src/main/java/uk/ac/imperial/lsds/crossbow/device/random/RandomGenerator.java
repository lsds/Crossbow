package uk.ac.imperial.lsds.crossbow.device.random;

import java.nio.ByteBuffer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;

public class RandomGenerator {
	
	private final static Logger log = LogManager.getLogger (RandomGenerator.class);
	
	private static final RandomGenerator randomInstance = new RandomGenerator ();
	
	public static RandomGenerator getInstance () { return randomInstance; }
	
	private boolean loaded;
	
	public RandomGenerator () {
		loaded = false;
	}
	
	public boolean isLoaded () {
		return loaded;
	}
	
	public void load () {
		
		if (! isLoaded()) {
			try {
				String library = String.format("%s/clib-multigpu/libRNG.so", SystemConf.getInstance().getHomeDirectory());
				System.load (library);
			} catch (final UnsatisfiedLinkError e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			loaded = true;
		}
	}
	
	public void randomUniformFill (IDataBuffer buffer, int count, float start, float end) {
		
		if (! isLoaded())
			throw new IllegalStateException ("error: random generator library is not loaded");
		
		if (! buffer.isDirect())
			throw new IllegalStateException ("error: random generator library operates only on direct buffers");
		
		if (! buffer.getType().isFloat())
			throw new IllegalStateException ("error: random generator library operates only on float buffers");
		
		randomUniformFill (buffer.getByteBuffer(), count, start, end);
	}
	
	public void randomGaussianFill (IDataBuffer buffer, int count, float mean, float std, int truncate) {
		
		if (! isLoaded())
			throw new IllegalStateException ("error: random generator library is not loaded");
		
		if (! buffer.isDirect())
			throw new IllegalStateException ("error: random generator library operates only on direct buffers");
		
		if (! buffer.getType().isFloat())
			throw new IllegalStateException ("error: random generator library operates only on float buffers");
		
		randomGaussianFill (buffer.getByteBuffer(), count, mean, std, truncate);
	}
	
	public native int test ();
	
	public native int init (long seed);
	
	public native int randomUniformFill  (ByteBuffer buffer, int count, float start, float end);
	public native int randomGaussianFill (ByteBuffer buffer, int count, float mean,  float std, int truncate);
	
	public native int destroy ();
}
