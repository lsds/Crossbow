package uk.ac.imperial.lsds.crossbow.utils;

import java.lang.reflect.Field;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import sun.misc.Unsafe;

@SuppressWarnings("restriction")
abstract class SlottedObjectPoolPad {

	protected long p1, p2, p3, p4, p5, p6, p7;
}

@SuppressWarnings("restriction")
public class SlottedObjectPool<T> extends SlottedObjectPoolPad {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (SlottedObjectPool.class);
	
	private static final int  pad;
	private static final long base;
	private static final int  shift;
	
	private static final Unsafe theUnsafe;
	
	static {
		
		try {
			Field unsafe = Unsafe.class.getDeclaredField("theUnsafe");
			unsafe.setAccessible(true);
			theUnsafe = (Unsafe) unsafe.get (null);
		} catch (Exception e) {
			throw new AssertionError(e);
		}
		
		final int scale = theUnsafe.arrayIndexScale (Object[].class);
		if (scale == 4) { 
			shift = 2;
		} else 
		if (scale == 8) { 
			shift = 3;
		} else {
			throw new IllegalStateException("Unknown pointer size");
		}
		
		pad = 128 / scale;
		
		base = theUnsafe.arrayBaseOffset(Object[].class) + (pad << shift);
	}
	
	private final int mask;
	
	private final Object [] entries;
	
	protected int size, elements;
	
	public SlottedObjectPool (int size) {
		
		this (size, null);
	}
	
	public SlottedObjectPool (int size, ObjectFactory<T> factory) {
		
		this.elements = size;
		
		this.size = size;
		while (Integer.bitCount(this.size) != 1)
			this.size ++;
		
		if (size < 1) {
			throw new IllegalArgumentException("error: ring size must greater than 0");
		}
		
		if (Integer.bitCount(this.size) != 1) {
			throw new IllegalArgumentException("error: ring size must be a power of 2");
		}
		
		/*
		log.debug(String.format("Buffer pad is %d", pad));
		log.debug(String.format("Array base offset is %d", base));
		log.debug(String.format("Element shift is %d", shift));
		*/
		
		mask = this.size - 1;
		
		entries = new Object [this.size + 2 * pad];
		
		fill(factory);
	}
	
	private void fill (ObjectFactory<T> factory) {
		
		for (int i = 0; i < size; i++) {
			
			entries[pad + i] = (factory != null) ? factory.newInstance(i) : null;
		}
	}
	
	@SuppressWarnings("unchecked")
	public final T elementAt (int offset) {
		
		return (T) theUnsafe.getObject(entries, base + ((offset & mask) << shift));
	}
	
	public void setElementAt (int offset, T element) {
		
		entries[pad + (offset & mask)] = element;
	}
	
	public int elements () {
		
		return elements;
	}
	
	public String toString () {
		
		StringBuilder b = new StringBuilder();
		
		b.append("SlottedObjectPool { ");
		b.append(String.format("%d entries", size));
		b.append("}");
		
		return b.toString();
	}
}
