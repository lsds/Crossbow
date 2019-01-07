package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;
import java.util.ArrayList;

public class DatasetFileStreamWriter {

private ArrayList<DatasetFileStream> outputs;
	
	private String prefix;
	
	private int limit;
	private int count;
	
	private int writes, tuplesWritten;
	
	public DatasetFileStreamWriter (String prefix) {
		
		this.prefix = prefix;
		
		limit = count = writes = tuplesWritten = 0;
		
		outputs = new ArrayList<DatasetFileStream>();
	}
	
	public void setLimit (int limit) {
		
		this.limit = limit;
	}
	
	public void setLimit (int limit, BatchDescriptor batch) {
		
		setLimit (limit * batch.getBatchSize());
	}
	
	private static int index (int count) {
		
		return (count - 1);
	}
	
	private static String getFilename (String prefix, int count) {
		
		return String.format("%s.%d", prefix, count);
	}
	
	private DatasetFileStream newInstance () throws IOException, IllegalStateException {
		
		if (limit == 0)
			throw new IllegalStateException ("error: limit is not set");
		
		DatasetFileStream next = new DatasetFileStream (getFilename (prefix, ++count), limit);
		
		outputs.add(next);
		
		return next;
	}
	
	private DatasetFileStream getOutputFileStream () throws IOException {
		
		if (outputs.isEmpty())
		{
			return newInstance ();
		}
		else
		{
			DatasetFileStream current = outputs.get(index (count));
			if (! current.isFull ())
			{
				return current;
			}
			else
			{
				current.close ();
				return newInstance ();
			}
		}
	}
	
	public DatasetFileStream getCurrentFileStream () {
		
		return outputs.get(index (count));
	}
	
	public void write (DataTuple tuple) throws IOException {
		
		DatasetFileStream stream = getOutputFileStream ();
		stream.write (tuple.getBuffer().array());
		writes ++;
		tuplesWritten ++;
		return;
	}
	
	public void fill (int pad) throws IOException {
		if (pad > 0) {
			DatasetFileStream stream = getOutputFileStream ();
			stream.write(pad);
		}
		writes = 0;
		return;
	}
	
	public void repeat (int missing, DataTuple tuple, DatasetFileStream stream) throws IOException {
		
		byte [] buffer = tuple.getBuffer().array();
		
		DatasetFileStream current = outputs.get (index (count));
		
		for (int i = 0; i < missing; ++i) {	
			stream.read(buffer);
			current.write(buffer);
		}
		
		writes += missing;
		tuplesWritten += missing;
	}
	
	public int counter () {
		
		return writes;
	}
	
	public int numberOfTuplesWritten () {
		
		return tuplesWritten;
	}
	
	public int numberOfFilesWritten () {
		
		return outputs.size();
	}
	
	public void close () throws IOException {
		
		if (outputs.isEmpty ())
			return;
		
		DatasetFileStream current = outputs.get (index (count));
		
		current.truncate ();
		current.close ();
	}
	
	void dump () {
		
		StringBuilder s = new StringBuilder ();
		
		s.append (String.format("=== [Dataset file stream writer: %d files, %d bytes] ===\n", outputs.size(), limit * outputs.size()));
		
		for (DatasetFileStream f: outputs) 
			s.append(f.toString()).append("\n");
		
		s.append("=== [End of dataset file stream writer dump] ===");
		
		System.out.println(s.toString());
	}
}
