package uk.ac.imperial.lsds.crossbow.utils;

import java.util.Arrays;
import java.util.Iterator;

public class CrossbowArrayList<T> implements Iterable<T> {
	
	Object [] elements;
	int size;
	
	public CrossbowArrayList () {
		elements = null;
		size = 0;
	}
	
	public boolean isEmpty () {
		return (size == 0);
	}
	
	public int size () {
		return size;
	}
	
	private void ensureCapacity () {
		if (size == 0) {
			elements = new Object [1];
		}
		else {
			if (size < elements.length)
				return;
			Object [] _elements = Arrays.copyOf(elements, size + 1);
			elements = _elements;
		}
	}
	
	public void set (int ndx, T element) {
		
		if (ndx < 0 || ndx > (size - 1))
			throw new ArrayIndexOutOfBoundsException();
		
		elements[ndx] = element;
	}
	
	public void add (T [] elements) {
		for (int i = 0; i < elements.length; ++i)
			append(elements[i]);
	}
	
	public void add (T element) {
		append (element);
	}
	
	public void append (T element) {
		ensureCapacity ();
		elements[size++] = element;
		return;
	}
	
	@SuppressWarnings("unchecked")
	public T get (int ndx) {
		
		if (ndx < 0 || ndx > (size - 1))
			throw new ArrayIndexOutOfBoundsException();
		
		return (T) elements [ndx];
	}
	
	@SuppressWarnings({ "unchecked", "hiding" })
	public <T> T [] array () {
		return (T []) elements;
	}
	
	@SuppressWarnings("unchecked")
	public CrossbowArrayList<T> copy () {
		CrossbowArrayList<T> list = new CrossbowArrayList<T>();
		list.add((T []) array());
		return list;
	}
	
	public Iterator<T> iterator () {
		return new Itr ();
	}
	
	private class Itr implements Iterator<T> {
		
		private int cursor;
		
		public Itr () {
			cursor = 0;
		}
		
		public boolean hasNext () {
			return (cursor != size);
		}
		
		@SuppressWarnings("unchecked")
		public T next () {
			return (T) elements[cursor++];
		}
		
		public void remove () {
			throw new IllegalStateException ();
		}
	}
}
