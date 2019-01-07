package uk.ac.imperial.lsds.crossbow.types;

public enum DurationUnit {
	
	HOURS (0), MINUTES (1), SECONDS (2);
	
	private int id;
	
	DurationUnit (int id) {
		this.id = id;
	}
	
	public String toString () {
		return toString (true);
	}
	
	public String toString (boolean plural) {
		switch (id) {
		case 0: return ((plural) ? "hours"    :   "hour");
		case 1: return ((plural) ? "minutes"  : "minute");
		case 2: return ((plural) ? "seconds"  : "second");
		default:
			throw new IllegalArgumentException ("error: invalid duration unit");
		}
	}

	public static DurationUnit fromString (String unit) {
		if (unit.toLowerCase().equals("hours"  )) return HOURS;
		else 
		if (unit.toLowerCase().equals("minutes")) return MINUTES;
		else 
		if (unit.toLowerCase().equals("seconds")) return SECONDS;
		else
			throw new IllegalArgumentException (String.format("error: invalid duration unit %s", unit));
	}
}
