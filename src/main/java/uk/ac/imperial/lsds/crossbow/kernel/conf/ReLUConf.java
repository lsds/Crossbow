package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.types.ActivationMode;

public class ReLUConf implements IConf {
	
	/* [From caffe, caffe.proto (l. 870)]
	 * 
	 * Allow non-zero slope for negative inputs to speed up optimization
	 */
	private float negativeSlope;
	
	private ActivationMode mode;
	
	double ceiling;
	  
	public ReLUConf () {
		negativeSlope = 0;
		ceiling = 0.0D;
		mode = ActivationMode.RELU;
	}
	
	public ReLUConf setNegativeSlope (float negativeSlope) {
		this.negativeSlope = negativeSlope;
		return this;
	}
	
	public float getNegativeSlope () {
		return negativeSlope;
	}
	
	public ReLUConf setReLUCeiling (double ceiling) {
		this.ceiling = ceiling;
		return this;
	}
	
	public double getReLUCeiling () {
		return ceiling;
	}
	
	public ReLUConf setActivationMode (ActivationMode mode) {
		this.mode = mode;
		return this;
	}
	
	public ActivationMode getActivationMode () {
		return mode;
	}
}
