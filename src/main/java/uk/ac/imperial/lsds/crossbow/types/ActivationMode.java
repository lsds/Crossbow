package uk.ac.imperial.lsds.crossbow.types;

public enum ActivationMode {
	
	SIGMOID(0), RELU(1), TANH(2), CLIPPEDRELU(3);
	
	private int id;
	
	ActivationMode (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "SIGMOID";
		case 1: return "RELU";
		case 2: return "TANH";
		case 3: return "CLIPPEDRELU";
		default:
			throw new IllegalArgumentException ("error: invalid activation mode");
		}
	}
}
