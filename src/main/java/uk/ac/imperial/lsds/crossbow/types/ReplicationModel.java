package uk.ac.imperial.lsds.crossbow.types;

public enum ReplicationModel {
	
	SRSW, MRSW;

	public static ReplicationModel fromString (String model) {
		
		if      (model.toUpperCase().equals("SRSW")) return SRSW;
		else if (model.toUpperCase().equals("MRSW")) return MRSW;
		else
			throw new IllegalArgumentException (String.format("error: invalid replication model: %s", model));
	}
}
