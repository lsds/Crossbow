package uk.ac.imperial.lsds.crossbow.types;

public enum TrainingUnit {
	
	EPOCHS (0), TASKS (1);
	
	private int id;
	
	TrainingUnit (int id) {
		this.id = id;
	}
	
	public String toString () {
		return toString (true);
	}
	
	public String toString (boolean plural) {
		switch (id) {
		case 0: return ((plural) ? "epochs" : "epoch");
		case 1: return ((plural) ? "tasks"  : "task" );
		default:
			throw new IllegalArgumentException ("error: invalid training unit");
		}
	}

	public static TrainingUnit fromString (String unit) {
		if (unit.toLowerCase().equals("epochs")) return EPOCHS;
		else 
		if (unit.toLowerCase().equals("tasks" )) return TASKS;
		else
			throw new IllegalArgumentException (String.format("error: invalid training unit: %s", unit));
	}
}
