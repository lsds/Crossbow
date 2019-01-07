package uk.ac.imperial.lsds.crossbow.types;

public enum BatchNormEstimatedMeanAndVarianceType {
	
	FIXED(0), CMA(1);
	
	private int id;
	
	BatchNormEstimatedMeanAndVarianceType (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "FIXED";
		case 1: return "CMA";
		default:
			throw new IllegalArgumentException ("error: invalid batch normalisation running mean type");
		}
	}
}
