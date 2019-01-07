package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;

public class DatasetFileStream {
	
	private File file;
	
	private FileInputStream input;
	private FileOutputStream output;
	
	private FileChannel channel;
	
	private long size; /* File size in bytes */
	
	public DatasetFileStream (String filename) throws IOException {

		file = new File(filename);
		
		input = new FileInputStream (file);
		output = null;
		
		channel = input.getChannel();
		
		size = channel.size();
	}
	
	public DatasetFileStream (String filename, long bytes) throws IOException {

		file = new File(filename);

		/* Create new output file */

		if (file.exists()) {
			file.delete();
			file.createNewFile();
		}
		
		input = null;
		output = new FileOutputStream (file);
		
		channel = output.getChannel();
		
		size = bytes;
	}
	
	public long getSize () {

		return size;
	}
	
	public double progress () throws IOException {
		
		return (((double) channel.position() / (double) size) * 100D);
	}

	public void truncate () throws IOException {

		channel.truncate(channel.position());
	}

	public void flush () throws IOException {
		
		if (output == null)
			return;
		
		channel.force(true);
		output.getFD().sync();
		
		return;
	}

	public void close () throws IOException {
		
		if (output != null)
			flush ();
		
		channel.close();
		
		if (output != null)
			output.close();
		
		if (input != null)
			input.close();
	}

	public boolean isFull () throws IOException {

		return (channel.position() == size);
	}
	
	public boolean hasRemaining () throws IOException {
		
		return (channel.position() < size);
	}
	
	public long remaining () throws IOException {
		
		return (size - channel.position());
	}
	
	public String toString () {
		
		double p = 0D;
		
		try { 
			p = progress (); 
		} catch (IOException e) { 
			e.printStackTrace ();
		}
		
		return String.format("%s: %d bytes, %6.2f%%", file.getName(), size, p);
	}

	public String getFilename () {

		return file.getName ();
	}
	
	public void read (byte [] buffer) throws IOException {
		int offset = 0;
		int remaining = buffer.length;
		int count;
		while (remaining > 0) {
			count = input.read (buffer, offset, remaining);
			offset += count;
			remaining -= count;
		}
		return;
	}
	
	public void write (byte [] buffer) throws IOException {
		
		output.write (buffer);
	}
	
	public void write (int pad) throws IOException {
		
		for (int i = 0; i < pad; ++i)
			output.write((byte) 0);
	}

	public void skip (long bytes) throws IOException {
		
		input.skip (bytes);
	}
}
