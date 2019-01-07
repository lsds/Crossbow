package uk.ac.imperial.lsds.crossbow.preprocess.cifar100;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTupleIterator;
import uk.ac.imperial.lsds.crossbow.preprocess.Encoder;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;

public class Cifar100Encoder extends Encoder {
	
	private boolean computeMeanImage;
	
	private String meanImageFilename = null;
	
	private int padding = 0;
	
	private int tupleCount, expectedTupleCount;
	
	private int percent_ = 0, _percent = 0;
	long _t, t_;
	long bytes;
	double mbps;
	
	private ByteBuffer meanImage;

	public Cifar100Encoder (EncoderConf conf) {
		
		super(conf);
	}
	
	public Cifar100Encoder setComputeMeanImage (boolean computeMeanImage) { 
		this.computeMeanImage = computeMeanImage;
		return this;
	}
	
	public Cifar100Encoder setExpectedCount (int expectedTupleCount) { 
		this.expectedTupleCount = expectedTupleCount;
		return this;
	}
	
	public Cifar100Encoder setMeanImageFilename (String meanImageFilename) {
		this.meanImageFilename = meanImageFilename;
		return this;
	}
	
	public Cifar100Encoder setPadding (int padding) {
		this.padding = padding;
		return this;
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
				
				if (computeMeanImage) {
					
					if (meanImage == null)
						meanImage = ByteBuffer.allocate(example.size()).order(ByteOrder.LITTLE_ENDIAN);
				}
				
				int _coarse_label = input.get () & 0xFF;
				int   _fine_label = input.get () & 0xFF;
				// System.out.println("Label is " + _label);
				label.getBuffer ().putInt (_fine_label);
				
				/*
				 * Parsing the Cifar-10 dataset without padding
				 * 
				 * int pixels = example.getShape ().countAllElements ();
				 * 
				 * for (int i = 0; i < pixels; ++i) {
				 * 	float pixel = (float) (input.get () & 0xFF);
				 * 	example.getBuffer ().putFloat (pixel * conf.getScaleFactor ());
				 * 	if (computeMeanImage) {
				 * 		int offset = i * example.getDataType().sizeOf();
				 * 		meanImage.putFloat (offset, meanImage.getFloat(offset) + pixel);
				 * 	}
				 * }
				 */
				
				/* Reset output buffer */
				for (int index = 0; index < example.getBuffer().capacity(); index += example.getDataType().sizeOf())
					example.getBuffer().putFloat(index, 0);
				
				int width  = example.getShape().get(1);
				int height = example.getShape().get(2);
				
				for (int c = 0; c < 3; c ++) {
					for (int y = 0; y < 32; y ++) {
						for (int x = 0; x < 32; x ++) {
							float pixel = (float) (input.get () & 0xFF);
							/* Transform pixel */
							float transformed = pixel * conf.getScaleFactor ();
							/* Transform pixel (custom): 
							 * 
							 * a) Assert that pixel is in the range [0, 255].
							 * b) Rescale to [ 0, 2].
							 * c) Rescale to [-1, 1].
							 */
							if ((transformed < 0) || (transformed > 255)) {
								System.err.println("error: pixel value is not in [0, 255]");
								System.exit(1);
							}
							transformed = (transformed / 127.5F) - 1;
							
							int x_ = x + padding;
							int y_ = y + padding;
							int offset = (c * (height * width) + y_ * width + x_) * example.getDataType().sizeOf();
							example.getBuffer ().putFloat (offset, transformed);
							if (computeMeanImage)
								meanImage.putFloat (offset, meanImage.getFloat(offset) + pixel);
						}
					}
				}
				/* Set output buffer position (it will be rewinded afterwards) */
				example.getBuffer().position (example.size());
				
				if (tupleCount < 0)
					_t = System.nanoTime();
				
				/* Inc. tuples processed */
				tupleCount ++;
				
				/* Inc. bytes processed */
				bytes += (long) example.size();
				
				/* Print statistics */
				if (expectedTupleCount > 0) {
					percent_ = (tupleCount * 100) / expectedTupleCount;
					if (percent_ == (_percent + 1)) {
						
						t_ = System.nanoTime();
						
						mbps = (bytes / 1048576.) / ((double) (t_ - _t) / 1000000000.);
						
						System.out.print(String.format("Pre-processing Cifar-100...%3d%% (%6.2f MB/s)\r", percent_, mbps));
						
						_percent = percent_;
						_t = t_;
						
						/* Reset byte counter */
						bytes = 0;
					}
				}
			}
		};
		
		return it;
	}
	
	public void computeAndStoreMeanImage () {
		
		if (! computeMeanImage)
			return;
		
		for (int index = 0; index < meanImage.capacity(); index += 4) {
			meanImage.putFloat (index, (meanImage.getFloat(index) / tupleCount));
		}
		
		/* Reset meanImage pointers (position) */
		meanImage.clear();
		
		File file = new File (meanImageFilename);
		
		try {
			
			@SuppressWarnings("resource")
			FileChannel channel = new RandomAccessFile (file, "rws").getChannel();
			
			MappedByteBuffer output = channel.map(FileChannel.MapMode.READ_WRITE, 0, meanImage.capacity());
			output.order(ByteOrder.LITTLE_ENDIAN);
			
			output.put(meanImage);
			
			channel.close();

		} catch (Exception e) {
			
			System.err.println("error: failed to write mean image file: " + e.getMessage());
			System.exit(1);
		}
	}
}
