package uk.ac.imperial.lsds.crossbow.types;

public enum DatasetType {
	
	BASIC(0), LIGHT(1), RECORD(2);
	
	private int id;
	
	DatasetType (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return  "BASIC";
		case 1: return  "LIGHT";
		case 2: return "RECORD";
		default:
			throw new IllegalArgumentException ("error: invalid dataset type");
		}
	}
}
