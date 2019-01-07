package uk.ac.imperial.lsds.crossbow.types;

public enum DataType {
	
	INT (0), FLOAT (1);
	
	private int id;
	
	DataType (int id) {
		
		this.id = id;
	}
	
	public int sizeOf () {
		
		return 4;
	}
	
	public String toString () {
		
		switch (id) {
		case 0: return   "int";
		case 1: return "float";
		default:
			throw new IllegalArgumentException ("error: invalid data type");
		}
	}

	public boolean isInt () {
		
		return (id == 0);
	}
	
	public boolean isFloat () {
		
		return (id == 1);
	}
	
	public static DataType fromString (String type) {
		
		if ("int".equals(type)) {
			return INT;
		}
		else if ("float".equals(type)) {
			return FLOAT;
		}
		else {
			throw new IllegalArgumentException (String.format("error: invalid data type: %s", type));
		}
	}
}
