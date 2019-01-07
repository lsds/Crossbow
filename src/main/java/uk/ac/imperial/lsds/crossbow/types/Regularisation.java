package uk.ac.imperial.lsds.crossbow.types;

public enum Regularisation {
	
	L1, L2;

	public static Regularisation fromString (String type) {
		
		if      (type.toUpperCase().equals("L1")) return L1;
		else if (type.toUpperCase().equals("L2")) return L2;
		else
			throw new IllegalArgumentException (String.format("error: invalid regularisation type: %s", type));
	}
}
