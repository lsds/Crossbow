package uk.ac.imperial.lsds.crossbow.preprocess;

import uk.ac.imperial.lsds.crossbow.SystemConf;

public class BatchDescriptor {
	
	private int count;
	
	private int bytes, pad;
	
	private boolean initialised;
	
	public BatchDescriptor (int count) {
		
		this.count = count;
		
		bytes = pad = 0;
		
		initialised = false;
	}
	
	public void init (DataTuple tuple, boolean padding) {
		
		if (isInitialised())
			return;
		
		bytes = count * tuple.size();
		
		/* Align batch to page size */
		
		pad = 0;
		
		if (padding) {
			while ((bytes + pad) % SystemConf.getInstance().getPageSize() != 0)
				pad++;
		}
		
		initialised = true;
		
		return;
	}
	
	public boolean isInitialised () {
		
		return initialised;
	}
	
	public int elements () {
		
		return count;
	}
	
	public int getBytes () {
		
		return bytes;
	}
	
	public int getPad () {
		
		return pad;
	}
	
	public int getBatchSize () {
		
		return (bytes + pad);
	}
}
