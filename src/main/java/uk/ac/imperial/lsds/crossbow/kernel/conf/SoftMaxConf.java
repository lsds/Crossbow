package uk.ac.imperial.lsds.crossbow.kernel.conf;

public class SoftMaxConf implements IConf {
	
	/* The axis along which to perform SoftMax */
	private int axis;
	
	public SoftMaxConf () {
		axis = 1;
	}
	
	public SoftMaxConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
	
	public int getAxis () {
		return axis;
	}
}
