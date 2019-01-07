package uk.ac.imperial.lsds.crossbow.microbenchmarks.slots;

public class Example {
	
	private int id;
	
	public Example () {
		this(0);
	}
	
	public Example (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
}
