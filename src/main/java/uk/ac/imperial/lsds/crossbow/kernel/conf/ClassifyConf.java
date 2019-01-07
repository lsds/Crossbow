package uk.ac.imperial.lsds.crossbow.kernel.conf;

public class ClassifyConf implements IConf {
	
	private int axis;
	
	public ClassifyConf () {
		axis = 1;
	}
	
	public ClassifyConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
	
	public int getAxis () {
		return axis;
	}
}
