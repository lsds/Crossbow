package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;

public class DatasetDescriptor {
	
	private String [] source;
	
	private String destination;
	
	private DatasetFileReader reader;
	private DatasetFileWriter writer;
	
	public DatasetDescriptor () {
		
		source = null;
		destination = null;
	}
	
	public DatasetDescriptor setSource (String [] source) throws IOException {
		
		this.source = source;
		
		reader = new DatasetFileReader (this.source);
		
		return this;
	}
	
	public DatasetDescriptor setSource (String source) throws IOException {
		
		return setSource (new String [] { source });
	}
	
	public DatasetDescriptor setSource (String directory, String [] source) throws IOException {
		
		int N = source.length;
		
		String [] input = new String [N];
		
		for (int i = 0; i < N; ++i)
			input[i] = DatasetUtils.buildPath (directory, source[i], true);
		
		return setSource (input);
	}
	
	public DatasetDescriptor setSource (String directory, String source) throws IOException {
		
		String [] input = new String [] { DatasetUtils.buildPath (directory, source, true) };
		
		return setSource (input);
	}
	
	public String [] getSource () {
		
		return source;
	}
	
	public DatasetDescriptor setDestination (String destination) {
		
		this.destination = destination;
		
		writer = new DatasetFileWriter (destination);
		
		return this;
	}
	
	public DatasetDescriptor setDestination (String directory, String destination) {
		
		return setDestination (DatasetUtils.buildPath (directory, destination, false));
	}
	
	public String getDestination () {
		
		return destination;
	}
	
	/**
	 * Returns true is this descriptor shares the same source files with `descriptor`.
	 * 
	 * @return
	 */
	public boolean sharesInputWith (DatasetDescriptor descriptor) {
		
		String [] other = descriptor.getSource();
		
		if (source.length != other.length) return false;
		
		for (int i = 0; i < source.length; ++i)
			if (! source[i].equals(other[i]))
				return false;
		
		return true;
	}
	
	public DatasetFileReader getDatasetFileReader () {
		
		return reader;
	}
	
	public DatasetFileWriter getDatasetFileWriter () {
		
		return writer;
	}
}
