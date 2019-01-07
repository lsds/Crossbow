package uk.ac.imperial.lsds.crossbow.types;

public enum HandlerType {
	
	TASK(0), CALLBACK(1), DATASET(2);
	
	private int id;
	
	HandlerType (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
}
