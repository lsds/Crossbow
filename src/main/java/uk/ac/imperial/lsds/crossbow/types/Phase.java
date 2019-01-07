package uk.ac.imperial.lsds.crossbow.types;

public enum Phase {
	
	TRAIN(0), CHECK(1);
	
	private int id;
	
	Phase (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "training";
		case 1: return     "test";
		default:
			throw new IllegalArgumentException ("error: invalid phase type");
		}
	}
}
