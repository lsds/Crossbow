package uk.ac.imperial.lsds.crossbow.types;

public enum PoolMethod {
	
	MAX (0), AVERAGE (1), STOCHASTIC (2);
	
	private int id;
	
	PoolMethod (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
}
