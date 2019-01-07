package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.dataset.DatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class TestMappedDataBuffer {
	
	private final static Logger log = LogManager.getLogger (TestMappedDataBuffer.class);
	
	public static void main (String [] args) {
		
		try {
		
		DatasetMemoryManager.getInstance().init();
		
		String metadata = SystemConf.getInstance().getHomeDirectory() + "/data/mnist/b-064/mnist-train.metadata";
		
		Dataset dataset = new Dataset (metadata);
		dataset.setPhase(Phase.TRAIN);
		dataset.init();
		
		MappedDataBuffer b1 = dataset.getExamples (0);
		MappedDataBuffer b2 = dataset.getLabels (0);
		
		int imageSize = (int) Math.pow(28D, 2D) * DataType.FLOAT.sizeOf();
		
		log.debug(String.format("%d bytes per image", imageSize));
		log.debug(String.format("%d images in file", b1.capacity() / imageSize));
		
		int ndx = 1;
		
		int start = ndx * imageSize;
		int end = start + imageSize;
		
		int label = b2.getInt (ndx);
		
		int pos = start;
		
		System.out.println(String.format("===[Image: %d; label: %d]===", ndx, label));
		int cnt = 0;
		while (pos < end) {
				
			if ((cnt % 28) == 0)
				System.out.println();
				
			float v = b1.getFloat(pos);
			if (v == 0F)
				System.out.print(String.format("%5d ", 0));
			else
				System.out.print(String.format("%1.3f ", v));
			
			pos += 4;
			cnt++;
		}
		System.out.println();
		System.out.println(String.format("===[End of image dump (%d pixels)]===", cnt));
		
		} 
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		DatasetMemoryManager.getInstance().free();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
