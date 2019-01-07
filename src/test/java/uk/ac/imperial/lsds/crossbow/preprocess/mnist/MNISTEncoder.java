package uk.ac.imperial.lsds.crossbow.preprocess.mnist;

import java.nio.ByteBuffer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTupleIterator;
import uk.ac.imperial.lsds.crossbow.preprocess.Encoder;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;

public class MNISTEncoder extends Encoder {
	
	private final static Logger log = LogManager.getLogger (MNISTEncoder.class);

	public MNISTEncoder (EncoderConf conf) {
		
		super(conf);
	}
	
	public DataTupleIterator iterator () {
		
		DataTupleIterator it = new DataTupleIterator () {
			
			@Override
			public void parseExamplesFileHeader (ByteBuffer input) {
				
				int magic = input.getInt ();
				if (magic != 2051) {
					
					System.err.println ("error: invalid magic number");
					System.exit(1);
				}
				
				int count  = input.getInt();
				int height = input.getInt();
				int width  = input.getInt();
				
				int bytes = count * height * width;
				
				log.info (String.format("%d %d x %d images (%d bytes) in input buffer (%d bytes remaining)", 
						
						count, height, width, bytes, input.remaining()));
			}
			
			@Override
			public void nextExample (ByteBuffer input, DataTuple example) {
				
				int pixels = example.getShape ().countAllElements ();
				
				ByteBuffer output = example.getBuffer ();
				
				for (int i = 0; i < pixels; ++i)
					output.putFloat ((float) (input.get() & 0xFF) * conf.getScaleFactor());
			}
			
			@Override
			public void parseLabelsFileHeader (ByteBuffer input) {
				
				int magic = input.getInt ();
				if (magic != 2049) {
					
					System.err.println ("error: invalid magic number");
					System.exit (1);
				}
				
				int count = input.getInt ();
				
				log.info (String.format ("%d labels (%d bytes) in input buffer (%d bytes remaining)", 
						
						count, count, input.remaining ()));
			}
			
			@Override
			public void nextLabel (ByteBuffer input, DataTuple label) {
				
				ByteBuffer output = label.getBuffer ();
				int value = input.get () & 0xFF;
				output.putInt (value);
			}
			
			@Override
			public void nextTuple (ByteBuffer input, DataTuple example, DataTuple label) {
				
				return;
			}
		};
		
		return it;
	}
}
