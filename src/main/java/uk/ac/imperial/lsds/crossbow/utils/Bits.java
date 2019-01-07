package uk.ac.imperial.lsds.crossbow.utils;

import java.lang.reflect.Field;
import java.nio.ByteOrder;
import java.security.AccessController;

import sun.misc.Unsafe;

/*
 * Exposing java.nio.Bits functions to Crossbow 
 * to manage MappedDataBuffer objects
 */
@SuppressWarnings("restriction")
public class Bits {

	private static final Unsafe unsafe;

	private static final ByteOrder byteOrder;
	
	private static boolean unaligned;
	private static boolean unalignedKnown = false;

	private Bits () {

	}
	
	static {
		
		try {
			
			Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
			theUnsafe.setAccessible(true);
			unsafe = (Unsafe) theUnsafe.get (null);
		} 
		catch (Exception e) {
			throw new AssertionError(e);
		}
	}
	
	public static Unsafe unsafe() {
		return unsafe;
	}
	
	public static boolean unaligned () {

		if (unalignedKnown)
			return unaligned;

		String arch = AccessController.doPrivileged 
				(new sun.security.action.GetPropertyAction("os.arch"));

		unaligned = arch.equals("i386") || arch.equals("x86") || arch.equals("amd64");
		unalignedKnown = true;

		return unaligned;
	}

	public static ByteOrder byteOrder () {
		
		if (byteOrder == null)
			throw new IllegalStateException ();
		return byteOrder;
	}
	
	static {
		long a = unsafe.allocateMemory(8);
		try {
			unsafe.putLong(a, 0x0102030405060708L);
			byte b = unsafe.getByte(a);
			switch (b) {
			case 0x01: byteOrder = ByteOrder.BIG_ENDIAN;     break;
			case 0x08: byteOrder = ByteOrder.LITTLE_ENDIAN;  break;
			default:
				byteOrder = null;
			}
		} finally {
			unsafe.freeMemory(a);
		}
	}

	public static int swap (int x) {
		
		return ((x << 24) | ((x & 0x0000ff00) <<  8) | ((x & 0x00ff0000) >>> 8) | (x >>> 24));
	}
	
	public static long swap (long x) {
		
		return (((long) swap((int) x) << 32) | ((long) swap((int) (x >>> 32)) & 0xffffffffL));
	}

	private static byte _get(long address) {
		
		return unsafe.getByte(address);
	}

	private static int makeInt (byte b3, byte b2, byte b1, byte b0) {
		
		return (((b3 & 0xff) << 24) |
				((b2 & 0xff) << 16) |
				((b1 & 0xff) <<  8) |
				((b0 & 0xff) <<  0));
	}

	static int getIntL(long address) {
		
		return makeInt(
				_get(address + 3),
				_get(address + 2),
				_get(address + 1),
				_get(address + 0));
	}
	
	static int getIntB(long address) {
		
		return makeInt(
				_get(address + 0),
				_get(address + 1),
				_get(address + 2),
				_get(address + 3));
	}
	
	public static int getInt (long address, boolean bigEndian) {

		return (bigEndian ? getIntB (address) : getIntL (address));
	}

	static private long makeLong (
			byte b7, byte b6, byte b5, byte b4,
			byte b3, byte b2, byte b1, byte b0
			) {
		
		return ((((long) b7 & 0xff) << 56) |
				(((long) b6 & 0xff) << 48) |
				(((long) b5 & 0xff) << 40) |
				(((long) b4 & 0xff) << 32) |
				(((long) b3 & 0xff) << 24) |
				(((long) b2 & 0xff) << 16) |
				(((long) b1 & 0xff) <<  8) |
				(((long) b0 & 0xff) <<  0));
	}

	static long getLongL(long address) {
		
		return makeLong(
				_get(address + 7),
				_get(address + 6),
				_get(address + 5),
				_get(address + 4),
				_get(address + 3),
				_get(address + 2),
				_get(address + 1),
				_get(address + 0));
	}

	static long getLongB(long address) {
		
		return makeLong(
				_get(address + 0),
				_get(address + 1),
				_get(address + 2),
				_get(address + 3),
				_get(address + 4),
				_get(address + 5),
				_get(address + 6),
				_get(address + 7));
	}
	
	public static long getLong (long address, boolean bigEndian) {

		return (bigEndian ? getLongB (address) : getLongL (address));
	}

	static float getFloatL (long address) {
		
		return Float.intBitsToFloat(getIntL(address));
	}

	static float getFloatB (long address) {
		
		return Float.intBitsToFloat(getIntB(address));
	}


	public static float getFloat (long address, boolean bigEndian) {
		
		return (bigEndian ? getFloatB(address) : getFloatL(address));
	}

}
