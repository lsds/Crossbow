package uk.ac.imperial.lsds.crossbow.types;

public enum LearningRateDecayPolicy {
	
	FIXED(0), INV(1), STEP(2), MULTISTEP(3), LSR(4), EXP(5), CLR(6);
	
	private int id;
	
	LearningRateDecayPolicy (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "FIXED";
		case 1: return "INV";
		case 2: return "STEP";
		case 3: return "MULTISTEP";
		case 4: return "LSR";
		case 5: return "EXP";
		case 6: return "CLR";
		default:
			throw new IllegalArgumentException ("error: invalid synchronisation model type");
		}
	}

	public static LearningRateDecayPolicy fromString(String policy) {
		if      (policy.toUpperCase().equals("FIXED"))     return FIXED;
		else if (policy.toUpperCase().equals("INV"))       return INV;
		else if (policy.toUpperCase().equals("STEP"))      return STEP;
		else if (policy.toUpperCase().equals("MULTISTEP")) return MULTISTEP;
		else if (policy.toUpperCase().equals("LSR"))       return LSR;
		else if (policy.toUpperCase().equals("EXP"))       return EXP;
		else if (policy.toUpperCase().equals("CLR"))       return CLR;
		else
			throw new IllegalArgumentException (String.format("error: invalid learning rate decay policy: %s", policy));
	}
}
