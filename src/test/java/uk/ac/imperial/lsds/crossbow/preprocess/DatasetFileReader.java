package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;
import java.util.Iterator;

public class DatasetFileReader implements Iterable<DatasetFile> {
	
	private DatasetFile [] inputs;
	
	private long size;
	
	public DatasetFileReader (String [] filenames) throws IOException {
		
		inputs = new DatasetFile [filenames.length];
		
		for (int i = 0; i < inputs.length; ++i) {
			
			inputs[i] = new DatasetFile (filenames[i]);
			size += inputs[i].getSize();
		}
	}
	
	public long getSize () {
		
		return size;
	}
	
	public int getCount () {
		
		return inputs.length;
	}
	
	public DatasetFile getDatasetFile (int ndx) {
		
		if (ndx < 0 || ndx >= inputs.length)
			throw new ArrayIndexOutOfBoundsException ("error: invalid file index");
		
		return inputs [ndx];
	}
	
	public Iterator<DatasetFile> iterator () {
		
		Iterator<DatasetFile> it = new Iterator<DatasetFile>() {
			
			private int cursor = 0;
			
			public boolean hasNext () {
				
				return (cursor < getCount());
			}
			
			public DatasetFile next () {
				
				return inputs[cursor++];
			}

			public void remove () {}
		};
		
		return it;
	}
	
	void dump () {
		
		StringBuilder s = new StringBuilder ();
		
		s.append (String.format("=== [Dataset file reader: %d files, %d bytes] ===\n", inputs.length, size));
		
		for (int i = 0; i < inputs.length; ++i) s.append(inputs[i].toString()).append("\n");
		
		s.append("=== [End of dataset file reader dump] ===");
		
		System.out.println(s.toString());
	}
}
