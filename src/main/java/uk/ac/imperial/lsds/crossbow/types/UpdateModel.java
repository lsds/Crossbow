package uk.ac.imperial.lsds.crossbow.types;

public enum UpdateModel {
	
	DEFAULT(0), WORKER(1), EAMSGD(2), SYNCHRONOUSEAMSGD(3), DOWNPOUR(4);
	
	private int id;
	
	UpdateModel (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "DEFAULT";
		case 1: return "WORKER";
		case 2: return "EAMSGD";
		case 3: return "SYNCHRONOUSEAMSGD";
		case 4: return "DOWNPOUR";
		default:
			throw new IllegalArgumentException ("error: invalid update model type");
		}
	}

	public static UpdateModel fromString(String model) {
		
		if      (model.toUpperCase().equals("DEFAULT"))           return DEFAULT;
		else if (model.toUpperCase().equals("WORKER"))            return WORKER;
		else if (model.toUpperCase().equals("EAMSGD"))            return EAMSGD;
		else if (model.toUpperCase().equals("SYNCHRONOUSEAMSGD")) return SYNCHRONOUSEAMSGD;
		else if (model.toUpperCase().equals("DOWNPOUR"))          return DOWNPOUR;
		else
			throw new IllegalArgumentException (String.format("error: invalid update model: %s", model));
	}
}
