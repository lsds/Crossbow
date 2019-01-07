package uk.ac.imperial.lsds.crossbow.model;

import java.util.Iterator;

//public class VariableIterator {
public class VariableIterator implements Iterator <Variable> {
	
	Model mdl;
	int cursor;
	Variable currentVar;
	
	public VariableIterator (Model model) {
		mdl = model;
		cursor = 0;
		currentVar = mdl.getVariables()[cursor];
	}

	public boolean hasNext () {
		return (currentVar != null);
	}
	
	private Variable findNext () {		
		Variable result = null;
		
		if(currentVar.next != null) {
			result = currentVar.next;
		} else {
			cursor++;
			for(int ndx = cursor; ndx < mdl.getVariables().length; ndx++) {
				Variable p = mdl.getVariables()[ndx];
				if (p != null) {
					cursor = ndx;
					result = p;
					break;
				}
			}
		}
		
		return result;
	}
	
	public Variable next () {
		Variable result = currentVar;
		currentVar = findNext();
		
		return result;
	}
	
	
	public VariableIterator reset () {
		cursor = 0;
		
		for(int ndx = 0; ndx < mdl.getVariables().length; ndx++) {
			Variable p = mdl.getVariables()[ndx];
			if (p != null) {
				cursor = ndx;
				currentVar = p;
				break;
			}
		}
		
		return this;
	}

	@Override
	public void remove() {
		throw new IllegalStateException();
		
	}
}
