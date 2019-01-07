package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.types.NormalisationMode;

public class LossConf implements IConf {
		
	private int ignoredLabelValue;
	
	private NormalisationMode mode;
	
	public LossConf () {
		ignoredLabelValue = -1;
		mode = NormalisationMode.BATCH;
	}
	
	public int getIgnoredLabelValue () {
		return ignoredLabelValue;
	}

	public LossConf setIgnoredLabelValue (int ignoredLabelValue) {
		this.ignoredLabelValue = ignoredLabelValue;
		return this;
	}

	public NormalisationMode getMode () {
		return mode;
	}

	public LossConf setMode (NormalisationMode mode) {
		this.mode = mode;
		return this;
	}
}
