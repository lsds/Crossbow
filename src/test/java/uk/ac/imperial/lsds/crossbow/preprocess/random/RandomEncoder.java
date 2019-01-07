package uk.ac.imperial.lsds.crossbow.preprocess.random;

import java.nio.ByteBuffer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTupleIterator;
import uk.ac.imperial.lsds.crossbow.preprocess.Encoder;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;

public class RandomEncoder extends Encoder {
	
	private final static Logger log = LogManager.getLogger (RandomEncoder.class);

	public RandomEncoder(EncoderConf conf) {
		super(conf);
	}

	@Override
	public DataTupleIterator iterator() {
		
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
				
				log.info(String.format("Example buffer position %d capacity %d", example.getBuffer().position(), example.getBuffer().capacity()));
				log.info(String.format("Label   buffer position %d capacity %d",   label.getBuffer().position(),   label.getBuffer().capacity()));
				
				label.getBuffer ().putInt (0);
				
				int pixels = example.getShape ().countAllElements ();
				
				log.info(String.format("Randomly generate %d pixels", pixels));
				
				for (int i = 0; i < pixels; ++i) {
					float pixel = 0;
					example.getBuffer ().putFloat (pixel * conf.getScaleFactor ());
				}
				
				return;
			}
		};
		
		return it;
	}
}
