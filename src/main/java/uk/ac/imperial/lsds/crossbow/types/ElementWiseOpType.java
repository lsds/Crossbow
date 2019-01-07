package uk.ac.imperial.lsds.crossbow.types;

public enum ElementWiseOpType {
	
	PRODUCT (0), SUM (1), MAX (2);
	
	private int id;
	
	ElementWiseOpType (int id) {
		this.id = id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "product";
		case 1: return     "sum";
		case 2: return     "max";
		default:
			throw new IllegalArgumentException ("error: invalid element-wise operation type");
		}
	}
}
