package uk.ac.imperial.lsds.crossbow.types;

public enum ExecutionMode {
	
	CPU (0), GPU (1), HYBRID (2);
	
	private int id;
	
	ExecutionMode (int id) {
		this.id = id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "CPU-only";
		case 1: return "GPU-only";
		case 2: return   "hybrid";
		default:
			throw new IllegalArgumentException ("error: invalid execution mode");
		}
	}
}
