package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public class DatasetFileWriter {
	
	private ArrayList<DatasetFile> outputs;
	
	private String prefix;
	
	private int limit;
	private int count;
	
	private int writes, tuplesWritten;
	
	public DatasetFileWriter (String prefix) {
		
		this.prefix = prefix;
		
		limit = count = writes = tuplesWritten = 0;
		
		outputs = new ArrayList<DatasetFile>();
	}
	
	public void setLimit (int limit) {
		
		this.limit = limit;
	}
	
	public void setLimit (int limit, BatchDescriptor batch) {
		
		setLimit (limit * batch.getBatchSize());
	}
	
	public int getLimit () {
		
		return limit;
	}
	
	private static int index (int count) {
		
		return (count - 1);
	}
	
	private static String getFilename (String prefix, int count) {
		
		return String.format("%s.%d", prefix, count);
	}
	
	private DatasetFile newInstance () throws IOException, IllegalStateException {
		
		if (limit == 0)
			throw new IllegalStateException ("error: limit is not set");
		
		DatasetFile next = new DatasetFile (getFilename (prefix, ++count), limit);
		
		outputs.add(next);
		
		return next;
	}
	
	private DatasetFile getOutputFile () throws IOException {
		
		if (outputs.isEmpty())
		{
			return newInstance ();
		}
		else
		{
			DatasetFile current = outputs.get(index (count));
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
	
	public DatasetFile getCurrentFile () {
		
		return outputs.get(index (count));
	}
	
	public void write (DataTuple tuple) throws IOException {
		
		DatasetFile file = getOutputFile ();
		ByteBuffer buffer = tuple.getBuffer();
		buffer.flip();
		file.getByteBuffer().put(buffer);
		/* Prepare tuple buffer to accept the next tuple */
		buffer.rewind();
		
		writes ++;
		tuplesWritten ++;
		
		return;
	}
	
	public void fill (int pad) throws IOException {
		
		if (pad > 0) {
			DatasetFile file = getOutputFile ();
		
			ByteBuffer buffer = file.getByteBuffer();
		
			for (int i = 0; i < pad; ++i)
				buffer.put((byte) 0);
		}
		
		writes = 0;
		
		return;
	}
	
	/**
	 * Repeats the first `missing` tuples from the first output file 
	 * to the end of the current output file.
	 * 
	 * @param missing
	 * @param tuple
	 * @throws IOException 
	 */
	public void repeat (int missing, DataTuple tuple) throws IOException {
		
		int bytes = missing * tuple.size ();
		
		DatasetFile first = outputs.get(index (1));
		
		ByteBuffer repeated = first.getByteBuffer ().duplicate ();
		repeated.position (0).limit (bytes);
		
		DatasetFile current = outputs.get (index (count));
		
		current.getByteBuffer ().put (repeated);
		
		writes += missing;
		tuplesWritten += missing;
		
		return;
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
		
		DatasetFile current = outputs.get (index (count));
		
		current.truncate ();
		current.close ();
	}
	
	void dump () {
		
		StringBuilder s = new StringBuilder ();
		
		s.append (String.format("=== [Dataset file writer: %d files, %d bytes] ===\n", outputs.size(), limit * outputs.size()));
		
		for (DatasetFile f: outputs) 
			s.append(f.toString()).append("\n");
		
		s.append("=== [End of dataset file writer dump] ===");
		
		System.out.println(s.toString());
	}
}
