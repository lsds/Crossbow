package uk.ac.imperial.lsds.crossbow.types;

public enum DependencyType {
	
	START_BEFORE_START (0), END_BEFORE_START (1);
	
	private int id;
	
	DependencyType (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "START_BEFORE_START";
		case 1: return   "END_BEFORE_START";
		default:
			throw new IllegalArgumentException ("error: invalid dependency type");
		}
	}
}
