package uk.ac.imperial.lsds.crossbow.model;

import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.VarianceNormalisation;

public class InitialiserConf {
	
	private InitialiserType type;
	
	private float value;
	
	private float min, max, mean, std;
	
	private VarianceNormalisation norm;
	
	private int sparse;
	
	private boolean truncate;
	
	public InitialiserConf () {
		type = InitialiserType.CONSTANT;
		value = 0;
		min = 0;
		max = 1;
		mean = 0;
		std = 0;
		norm = VarianceNormalisation.FAN_IN;
		// norm = VarianceNormalisation.AVG;
		sparse = -1;
		truncate = false;
	}
	
	public InitialiserType getType () {
		return type;
	}

	public InitialiserConf setType (InitialiserType type) {
		this.type = type;
		return this;
	}
	
	public float getValue () {
		return value;
	}
	
	public InitialiserConf setValue (float value) {
		this.value = value;
		return this;
	}
	
	public boolean truncate () {
		return truncate;
	}
	
	public InitialiserConf truncate (boolean truncate) {
		this.truncate = truncate;
		return this;
	}
	
	public float getMin () {
		return min;
	}

	public InitialiserConf setMin (float min) {
		this.min = min;
		return this;
	}

	public float getMax () {
		return max;
	}

	public InitialiserConf setMax (float max) {
		this.max = max;
		return this;
	}

	public float getMean () {
		return mean;
	}

	public InitialiserConf setMean (float mean) {
		this.mean = mean;
		return this;
	}

	public float getStd () {
		return std;
	}

	public InitialiserConf setStd (float std) {
		this.std = std;
		return this;
	}

	public VarianceNormalisation getNorm () {
		return norm;
	}

	public InitialiserConf setNorm (VarianceNormalisation norm) {
		this.norm = norm;
		return this;
	}
	
	public int getSparse () {
		return sparse;
	}
	
	public InitialiserConf setSparse (int sparse) {
		this.sparse = sparse;
		return this;
	}
}
