package uk.ac.imperial.lsds.crossbow.model;

public class LocalVariable {
	
	private Variable [] theVariable;
	
	private ThreadLocal<Variable []> var;
	
	public LocalVariable (Variable ... v) {
		
		theVariable = v;
		
		var = new ThreadLocal<Variable []>() {
			
			protected Variable [] initialValue () {
				
				Variable [] variable = new Variable [theVariable.length];
				
				for (int i = 0; i < variable.length; ++i)
					variable[i] = theVariable[i].copy();
				
				return variable;
			}
		};
	}
	
	public Variable [] get () {
		return var.get();
	}
	
	public Variable [] getInitialValue () {
		return theVariable;
	}
}
