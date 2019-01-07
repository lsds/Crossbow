package uk.ac.imperial.lsds.crossbow.kernel.conf;

public class DropoutConf implements IConf {
	
	float ratio;
	  
	public DropoutConf () {
		ratio = 0.5F;
	}
	
	public DropoutConf setRatio (float ratio) {
		this.ratio = ratio;
		return this;
	}
	
	public float getRatio () {
		return ratio;
	}
}
