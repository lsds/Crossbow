package uk.ac.imperial.lsds.crossbow.types;

public enum SynchronisationModel {
	
	BSP(0), SSP(1), ASP(2);
	
	private int id;
	
	SynchronisationModel (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "BSP";
		case 1: return "SSP";
		case 2: return "ASP";
		default:
			throw new IllegalArgumentException ("error: invalid synchronisation model type");
		}
	}
	
	public static SynchronisationModel fromString (String model) {
		
		if      (model.toUpperCase().equals("BSP")) return BSP;
		else if (model.toUpperCase().equals("SSP")) return SSP;
		else if (model.toUpperCase().equals("ASP")) return ASP;
		else
			throw new IllegalArgumentException (String.format("error: invalid synchronisation model: %s", model));
	}
}
