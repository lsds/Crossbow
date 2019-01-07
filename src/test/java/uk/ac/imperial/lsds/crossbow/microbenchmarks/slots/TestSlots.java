package uk.ac.imperial.lsds.crossbow.microbenchmarks.slots;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

public class TestSlots {
	
	private final static Logger log = LogManager.getLogger (TestSlots.class);
	
	public static void main (String [] args) {
		
		ExampleFactory factory = new ExampleFactory();
		
		int ringSize = 32;
		
		SlottedObjectPool<Example> ring = new SlottedObjectPool<Example>(ringSize, factory);
		
		log.info(ring);
		
		for (int i = 0; i < ringSize; ++i) {
			
			Example e = ring.elementAt(i);
			log.info(String.format("Element at %2d is %2d", i, e.getId()));
		}
		
		ring.setElementAt(ringSize + 0, new Example(32));
		ring.setElementAt(ringSize + 1, new Example(33));
		
		log.info(String.format("Element at %2d is %2d", 0, ring.elementAt(0).getId()));
		log.info(String.format("Element at %2d is %2d", 1, ring.elementAt(1).getId()));
	}
}
