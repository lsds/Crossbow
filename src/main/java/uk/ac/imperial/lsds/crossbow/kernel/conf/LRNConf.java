package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.types.NormalisationRegion;

public class LRNConf implements IConf {
	
	/*
// Message that stores parameters used by LRNLayer
message LRNParameter {
  optional uint32 local_size = 1 [default = 5];
  optional float alpha = 2 [default = 1.];
  optional float beta = 3 [default = 0.75];
  enum NormRegion {
    ACROSS_CHANNELS = 0;
    WITHIN_CHANNEL = 1;
  }
  optional NormRegion norm_region = 4 [default = ACROSS_CHANNELS];
  optional float k = 5 [default = 1.];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 6 [default = DEFAULT];
}
	 */
	
	int size;
	
	NormalisationRegion region;
	
	float alpha, beta, kappa;
	  
	public LRNConf () {
		size = 5;
		region = NormalisationRegion.ACROSS_CHANNELS;
		alpha = 1F;
		beta = 0.75F;
		kappa = 1F;
	}
	
	public LRNConf setSize (int size) {
		this.size = size;
		return this;
	}
	
	public int getSize () {
		return size;
	}
	
	public LRNConf setNormalisationRegion (NormalisationRegion region) {
		this.region = region;
		return this;
	}
	
	public NormalisationRegion getNormalisationRegion () {
		return region;
	}
	
	public LRNConf setAlpha (float alpha) {
		this.alpha = alpha;
		return this;
	}
	
	public float getAlpha () {
		return alpha;
	}
	
	public LRNConf setBeta (float beta) {
		this.beta = beta;
		return this;
	}
	
	public float getBeta () {
		return beta;
	}
	
	public LRNConf setKappa (float kappa) {
		this.kappa = kappa;
		return this;
	}
	
	public float getKappa () {
		return kappa;
	}
}
