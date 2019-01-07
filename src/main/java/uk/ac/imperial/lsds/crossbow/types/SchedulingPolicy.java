package uk.ac.imperial.lsds.crossbow.types;

public enum SchedulingPolicy {
	
	HLS, FIFO, NULL;

	public static SchedulingPolicy fromString (String policy) {
		
		if      (policy.toUpperCase().equals("HLS"))  return  HLS;
		else if (policy.toUpperCase().equals("FIFO")) return FIFO;
		else if (policy.toUpperCase().equals("NULL")) return NULL;
		else
			throw new IllegalArgumentException (String.format("error: invalid scheduling policy: %s", policy));
	}
}
