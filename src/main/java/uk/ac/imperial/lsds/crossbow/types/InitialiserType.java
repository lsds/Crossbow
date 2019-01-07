package uk.ac.imperial.lsds.crossbow.types;

public enum InitialiserType {
	
	CONSTANT(0), UNIFORM(1), XAVIER(2), GAUSSIAN(3), MSRA(4);
	
	private int id;
	
	InitialiserType (int id) {
		this.id = id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "CONSTANT";
		case 1: return  "UNIFORM";
		case 2: return   "XAVIER";
		case 3: return "GAUSSIAN";
		case 4: return     "MSRA";
		default:
			throw new IllegalArgumentException ("error: invalid initialiser type");
		}
	}
	
	public static InitialiserType fromString (String type) {
			
		if      (type.toUpperCase().equals("CONSTANT")) return CONSTANT;
		else if (type.toUpperCase().equals("UNIFORM"))  return UNIFORM;
		else if (type.toUpperCase().equals("XAVIER"))   return XAVIER;
		else if (type.toUpperCase().equals("GAUSSIAN")) return GAUSSIAN;
		else if (type.toUpperCase().equals("MSRA"))     return MSRA;
		else
			throw new IllegalArgumentException (String.format("error: invalid initialiser type: %s", type));
	}
}
