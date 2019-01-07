package uk.ac.imperial.lsds.crossbow.preprocess;

public class BatchInfo {
	
	private int count;
	
	private BatchDescriptor example, label;
	
	public BatchInfo (int count) {
		
		this.count = count;
		
		example = new BatchDescriptor (count);
		  label = new BatchDescriptor (count);
	}
	
	public int elements () {
		
		return count;
	}
	
	public BatchDescriptor getExampleBatchDescriptor () {
		
		return example;
	}
	
	public BatchDescriptor getLabelBatchDescriptor () {
		
		return label;
	}
	
	public void init (DataTuplePair pair, boolean padding) {
		
		example.init (pair.getExample (), padding);
		  label.init (pair.getLabel (),   padding);
	}
	
	public long numberOfBatchesPerFile (long partitionSize) {
		
		/* Set batch size to the maximum value */
		
		long batchSize = (example.getBatchSize () > label.getBatchSize ()) ? example.getBatchSize () : label.getBatchSize ();
		
		if (partitionSize < batchSize) {
			
			System.err.println (String.format ("error: file partition size must be greater %d", batchSize));
			System.exit (1);
		}
		
		/* Align partition size to batch size */
		while ((partitionSize % batchSize) != 0)
			partitionSize--;
		
		return (partitionSize / batchSize);
	}
}
