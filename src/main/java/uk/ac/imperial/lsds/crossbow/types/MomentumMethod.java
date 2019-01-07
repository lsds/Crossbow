package uk.ac.imperial.lsds.crossbow.types;

public enum MomentumMethod {
	
	POLYAK(0), NESTEROV(1);
	
	private int id;
	
	MomentumMethod (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return   "POLYAK";
		case 1: return "NESTEROV";
		default:
			throw new IllegalArgumentException ("error: invalid momentum method");
		}
	}

	public static MomentumMethod fromString (String method) {
		if (method.toLowerCase().equals("polyak"  )) return POLYAK;
		else 
		if (method.toLowerCase().equals("nesterov")) return NESTEROV;
		else
			throw new IllegalArgumentException (String.format("error: invalid momentum method: %s", method));
	}
}
