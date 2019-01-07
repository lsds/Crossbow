package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;

public class DatasetFile {

	private File file;
	private RandomAccessFile randomAccessFile;

	private MapMode mode;
	private FileChannel channel;

	private long size; /* File size in bytes */

	private MappedByteBuffer buffer;

	public DatasetFile(String filename) throws IOException {

		file = new File(filename);

		randomAccessFile = new RandomAccessFile(file, "r");

		mode = MapMode.READ_ONLY;

		channel = randomAccessFile.getChannel();

		size = channel.size();

		buffer = null;
	}

	public DatasetFile(String filename, long bytes) throws IOException {

		file = new File(filename);

		/* Create new output file */

		if (file.exists()) {
			file.delete();
			file.createNewFile();
		}

		randomAccessFile = new RandomAccessFile(file, "rws");

		mode = MapMode.READ_WRITE;

		channel = randomAccessFile.getChannel();

		size = bytes;

		buffer = null;
	}

	public ByteBuffer getByteBuffer() throws IOException {

		if (!isLoaded())
			load();

		return buffer;
	}

	public long getSize() {

		return size;
	}

	public double progress() {

		if (!isLoaded())
			return 0D;

		return (((double) buffer.position() / (double) size) * 100D);
	}

	public boolean isLoaded() {

		return (buffer != null);
	}

	public void load() throws IOException {

		if (isLoaded()) {
			return;
		}

		buffer = channel.map(mode, 0, size); /* load(); */
		/*
		 * If the file is opened for writing, set byte order to LE.
		 */
		if (mode.equals(MapMode.READ_WRITE))
			buffer.order(ByteOrder.LITTLE_ENDIAN);
	}

	public void truncate() throws IOException {

		if (!isLoaded())
			return;

		channel.truncate((long) buffer.position());
	}

	public void flush() throws IOException {

		channel.force(true);
		randomAccessFile.getFD().sync();
	}

	public void close() throws IOException {

		/*
		 * Avoid this call until further notice...
		 * 
		 * MappedDataBufferCleaner.unmap(buffer);
		 */
		if (mode.equals(MapMode.READ_WRITE))
			flush();

		channel.close();
		randomAccessFile.close();
	}

	public boolean isFull() {

		if (!isLoaded())
			return false;

		return (! buffer.hasRemaining());
	}

	public String toString() {

		return String.format("%s: %d bytes, %6.2f%%", file.getName(), size,
				progress());
	}

	public String getFilename() {

		return file.getName();
	}
}
