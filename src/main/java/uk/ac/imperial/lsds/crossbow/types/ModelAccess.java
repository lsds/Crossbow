package uk.ac.imperial.lsds.crossbow.types;

public enum ModelAccess {
	
	NA(0), RO(1), RW(2);
	
	private int id;
	
	ModelAccess (int id) {
		this.id = id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "N/A";
		case 1: return "R/O";
		case 2: return "R/W";
		default:
			throw new IllegalArgumentException ("error: invalid model access type");
		}
	}
}
