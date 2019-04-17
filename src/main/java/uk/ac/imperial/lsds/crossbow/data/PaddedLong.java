package uk.ac.imperial.lsds.crossbow.data;

public class PaddedLong {
	
	public long value;
	
	public PaddedLong (long value) {
		
		this.value = value;
	}
	
	public volatile long _2, _3, _4, _5, _6, _7 = 7L;
	
	public long dummy () {
		
		return (_2 + _3 + _4 + _5 + _6 + _7);
	}
}

