package uk.ac.imperial.lsds.crossbow.preprocess.ratings;

import java.nio.ByteBuffer;

import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTupleIterator;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.preprocess.Encoder;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;

public class RatingsEncoder extends Encoder {

	public RatingsEncoder (EncoderConf conf) {
		
		super (conf);
	}
	
	public DataTupleIterator iterator () {
		
		DataTupleIterator it = new DataTupleIterator () {

			@Override
			public void parseExamplesFileHeader (ByteBuffer input) {
				
				return;
			}
			
			@Override
			public void nextExample (ByteBuffer input, DataTuple example) {
				
				return;
			}
			
			@Override
			public void parseLabelsFileHeader (ByteBuffer input) {
				
				return;
			}

			@Override
			public void nextLabel (ByteBuffer input, DataTuple label) {
				
				return;
			}

			@Override
			public void nextTuple (ByteBuffer input, DataTuple example, DataTuple label) {
				
				String line = DatasetUtils.readLine (input);
				String [] fields = line.split (" ");
				
				example.getBuffer ().putInt (Integer.parseInt (fields [0]));
				example.getBuffer ().putInt (Integer.parseInt (fields [1]));
				
				label.getBuffer ().putFloat (Float.parseFloat (fields [2]));
			}
		};
		
		return it;
	}
}
