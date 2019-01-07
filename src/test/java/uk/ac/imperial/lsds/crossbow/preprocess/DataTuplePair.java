package uk.ac.imperial.lsds.crossbow.preprocess;

public class DataTuplePair {
	
	private DataTuple example, label;
	
	public DataTuplePair () {
		
		this (null, null);
	}
	
	public DataTuplePair (DataTuple example, DataTuple label) {
		
		this.example = example;
		this.label = label;
	}
	
	public DataTuple getExample () {
		
		return example;
	}
	
	public DataTuple getLabel () {
		
		return label;
	}
}
