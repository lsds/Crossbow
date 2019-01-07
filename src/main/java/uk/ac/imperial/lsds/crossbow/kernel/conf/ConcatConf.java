package uk.ac.imperial.lsds.crossbow.kernel.conf;

public class ConcatConf implements IConf {
	
	/*
	 * The axis along which to concatenate input variables.
	 */
	private int axis;
	
	private int offset;
	
	public ConcatConf () {
		axis = 1;
		offset = 0;
	}
	
	public int getAxis () {
		return axis;
	}

	public ConcatConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
	
	public int getOffset () {
		return offset;
	}

	public ConcatConf setOffset (int offset) {
		this.offset = offset;
		return this;
	}
}
