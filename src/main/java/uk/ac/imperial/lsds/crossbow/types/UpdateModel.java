package uk.ac.imperial.lsds.crossbow.types;

public enum UpdateModel {
	
	DEFAULT(0), WORKER(1), EAMSGD(2), SYNCHRONOUSEAMSGD(3), DOWNPOUR(4), HOGWILD(5), POLYAK_RUPPERT(6), SMA(7);
	
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
		case 5: return "HOGWILD";
		case 6: return "POLYAK-RUPPERT";
		case 7: return "SMA";
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
		else if (model.toUpperCase().equals("HOGWILD"))           return HOGWILD;
		else if (model.toUpperCase().equals("POLYAK-RUPPERT"))    return POLYAK_RUPPERT;
		else if (model.toUpperCase().equals("SMA"))               return SMA;
		else
			throw new IllegalArgumentException (String.format("error: invalid update model: %s", model));
	}
}
