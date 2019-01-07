package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;
import java.nio.ByteBuffer;

public abstract class DataTupleIterator {
	
	public int __nextTuple (DatasetFile file, DataTuplePair pair) throws IOException {
		
		return __nextTuple (file, pair.getExample(), pair.getLabel());
	}
	
	public int __nextTuple (DatasetFile file, DataTuple example, DataTuple label) throws IOException {
		
		return __nextTuple (file.getByteBuffer(), example, label);
	}
	
	public int __nextTuple (ByteBuffer input, DataTuplePair pair) throws IOException {
		
		return __nextTuple (input, pair.getExample(), pair.getLabel());
	}
	
	public int __nextTuple (ByteBuffer input, DataTuple example, DataTuple label) {
		
		int pos = (input != null) ? input.position() : 0;
		
		nextTuple (input, example, label);
		
		return (input != null) ? (input.position() - pos) : 1;
	}
	
	public int __nextExample (ByteBuffer input, DataTuple example) {
		
		int pos = input.position();
		
		nextExample (input, example);
		
		return (input.position() - pos);
	}
	
	public int __nextLabel (ByteBuffer input, DataTuple label) {
		
		int pos = input.position();
		
		nextLabel (input, label);
		
		return (input.position() - pos);
	}
	
	/**
	 * 
	 * @param input
	 */
	public abstract void parseExamplesFileHeader (ByteBuffer input);
	
	/**
	 * 
	 * @param input
	 */
	public abstract void parseLabelsFileHeader (ByteBuffer input);
	
	/**
	 * 
	 * @param input
	 * @param example
	 */
	public abstract void nextExample (ByteBuffer input, DataTuple example);
	
	/**
	 * 
	 * @param input
	 * @param label
	 */
	public abstract void nextLabel (ByteBuffer input, DataTuple label);
	
	/**
	 * 
	 * @param input
	 * @param example
	 * @param label
	 */
	public abstract void nextTuple (ByteBuffer input, DataTuple example, DataTuple label);
}
