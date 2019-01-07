package uk.ac.imperial.lsds.crossbow.preprocess;

public class EncoderConf {
	
	private DatasetInfo dataset;
	
	private DataTuplePair pair;
	
	private BatchInfo info;
	
	private String metadata;
	
	private float scalefactor;
	
	/* Reshuffle options */
	private boolean copy;
	private String temp;
	
	/* Padding options */
	private boolean padding;
	
	public EncoderConf () {
		
		dataset = null;
		pair = null;
		info = null;
		metadata = null;
		
		scalefactor = 1F;
		
		copy = false;
		temp = null;
		
		padding = true;
	}
	
	public EncoderConf setDataset (DatasetInfo dataset) {
		
		this.dataset = dataset;
		
		return this;
	}
	
	public EncoderConf setDataset (DatasetDescriptor examples, DatasetDescriptor labels) {
		
		this.dataset = new DatasetInfo (examples, labels);
		
		return this;
	}
	
	public DatasetInfo getDataset () {
		
		return dataset;
	}
	
	public EncoderConf setDataTuplePair (DataTuple example, DataTuple label) {
		
		return setDataTuplePair (new DataTuplePair (example, label));
	}
	
	public EncoderConf setDataTuplePair (DataTuplePair pair) {
		
		this.pair = pair;
		
		if (info != null)
			info.init (pair, padding);
		
		return this;
	}
	
	public DataTuplePair getDataTuplePair () {
		
		return pair;
	}
	
	public EncoderConf setBatchInfo (int size) {
		
		return setBatchInfo (new BatchInfo (size));
	}
	
	public EncoderConf setBatchInfo (BatchInfo info) {
		
		this.info = info;
		
		if (pair != null)
			this.info.init (pair, padding);
		
		return this;
	}
	
	public BatchInfo getBatchInfo () {
		
		return info;
	}
	
	public EncoderConf setMetadata (String metadata) {
		
		this.metadata = metadata;
		
		return this;
	}
	
	public EncoderConf setMetadata (String directory, String filename) {
		
		return setMetadata (DatasetUtils.buildPath (directory, filename, false));
	}
	
	public String getMetadata () {
		
		return metadata;
	}
	
	public EncoderConf setScaleFactor (float scalefactor) {
		
		this.scalefactor = scalefactor;
		
		return this;
	}
	
	public float getScaleFactor () {
		
		return scalefactor;
	}
	
	public EncoderConf setCopyBeforeReshuffle (boolean copy) {
		
		this.copy = copy;
		
		return this;
	}
	
	public boolean copyBeforeReshuffle () {
		
		return copy;
	}
	
	public EncoderConf setTemporaryFile (String temp) {
		
		this.temp = temp;
		
		return this;
	}
	
	public String getTemporaryFile () {
		
		return temp;
	}
	
	public EncoderConf setPadding (boolean padding) {
		
		this.padding = padding;
		
		return this;
	}
	
	public boolean usePadding () {
		
		return padding;
	}
	
	public void configure (long partitionSize) {
		
		int limit = (int) info.numberOfBatchesPerFile (partitionSize);
		
		dataset.getExamplesDescriptor ().getDatasetFileWriter ().setLimit (limit, info.getExampleBatchDescriptor());
		dataset.getLabelsDescriptor   ().getDatasetFileWriter ().setLimit (limit, info.getLabelBatchDescriptor  ());
	}
}
