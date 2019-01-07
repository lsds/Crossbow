package uk.ac.imperial.lsds.crossbow.preprocess;

import java.util.Iterator;

public class DatasetInfo implements Iterable<DatasetDescriptor> {
	
	private DatasetDescriptor examples, labels;
	
	public DatasetInfo () {
		
		this (null, null);
	}
	
	public DatasetInfo (DatasetDescriptor examples, DatasetDescriptor labels) {
		
		this.examples = examples;
		this.labels = labels;
	}
	
	public DatasetInfo setExamplesDescriptor (DatasetDescriptor examples) {
		
		this.examples = examples;
		
		return this;
	}
	
	public DatasetDescriptor getExamplesDescriptor () {
		
		return examples;
	}
	
	public DatasetInfo setLabelsDescriptor (DatasetDescriptor labels) {
		
		this.labels = labels;
		
		return this;
	}
	
	public DatasetDescriptor getLabelsDescriptor () {
		
		return labels;
	}

	public Iterator<DatasetDescriptor> iterator () {
		
		Iterator<DatasetDescriptor> it = new Iterator<DatasetDescriptor>() {
			
			private int cursor = 2;
			
			public boolean hasNext () {
				return (cursor > 0);
			}
			
			public DatasetDescriptor next () {
				if (cursor-- > 1) 
					return examples;
				else 
					return labels;
			}

			public void remove () {}
		};
		
		return it;
	}

	public void configure(int limit, BatchInfo batch) {
		
		
	}
}
