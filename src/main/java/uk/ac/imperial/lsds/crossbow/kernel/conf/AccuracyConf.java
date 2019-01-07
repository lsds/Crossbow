package uk.ac.imperial.lsds.crossbow.kernel.conf;

public class AccuracyConf implements IConf {
	
	private int k;
	private int axis;
	private int ignoredLabelValue;
	
	public AccuracyConf () {
		k = 1;
		axis = 1;
		ignoredLabelValue = -1;
	}
	
	public AccuracyConf setTopK (int k) {
		this.k = k;
		return this;
	}
	
	public int getTopK () {
		return k;
	}
	
	public AccuracyConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
	
	public int getAxis () {
		return axis;
	}
	
	public AccuracyConf setIgnoredLabelValue (int ignoredLabelValue) {
		this.ignoredLabelValue = ignoredLabelValue;
		return this;
	}
	
	public int getIgnoredLabelValue () {
		return ignoredLabelValue;
	}
}
