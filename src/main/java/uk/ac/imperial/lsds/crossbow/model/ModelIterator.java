package uk.ac.imperial.lsds.crossbow.model;

import java.util.Iterator;
import uk.ac.imperial.lsds.crossbow.utils.Linked;

public class ModelIterator<T extends Linked <T>> implements Iterator<T> {
	T [] items;
	int cursor;
	T current;
	
	public ModelIterator (T [] items) {
		this.items = items;
		cursor = 0;
		current = items[cursor];
	}

	public boolean hasNext () {
		return (current != null);
	}
	
	private T findNext () {		
		T result = null;
		
		if(current.getNext() != null) {
			result = current.getNext();
		} else {
			cursor++;
			for(int ndx = cursor; ndx < items.length; ndx++) {
				T p = items[ndx];
				if (p != null) {
					cursor = ndx;
					result = p;
					break;
				}
			}
		}
		
		return result;
	}
	
	public T next () {
		T result = current;
		current = findNext();
		
		return result;
	}
	
	
	public ModelIterator<T> reset () {
		cursor = 0;
		
		for(int ndx = 0; ndx < items.length; ndx++) {
			T p = items[ndx];
			if (p != null) {
				cursor = ndx;
				current = p;
				break;
			}
		}
		
		return this;
	}

	@Override
	public void remove() {
		throw new IllegalStateException ();
	}
}
