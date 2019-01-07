package uk.ac.imperial.lsds.crossbow.kernel.conf;

public class NoopConf implements IConf {
	
	int axis;
	
	public NoopConf () {
		axis = 1;
	}
	
	public int getAxis () {
		return axis;
	}
	
	public NoopConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
}
