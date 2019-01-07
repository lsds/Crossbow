package uk.ac.imperial.lsds.crossbow.result;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.types.Phase;

public class ResultCollector implements Runnable {
	
	private final static Logger log = LogManager.getLogger (ResultCollector.class);
	
	private Phase phase;
	private IResultHandler handler;
	
	private CountDownLatch latch;
	
	private AtomicLong numberoftasks;
	
	public ResultCollector (Phase phase, IResultHandler handler) {
		this.phase = phase;
		this.handler = handler;
		/* Note that the latch is set after the collector has started */
		this.latch = null;
		
		numberoftasks = new AtomicLong(0L);
	}
	
	public long getNumberOfTasks () {
		
		return numberoftasks.get();
	}
	
	public void setCountDownLatch (CountDownLatch latch) {
		this.latch = latch;
	}
	
	public void run () {
		
		int next;
		
		while (true) {
			
			next = handler.getNext();
			
			/* Is slot `next` occupied? */
			while (! handler.ready (next)) {
				Thread.yield();
			}
			
			handler.freeSlot (next);
			
			if (log.isDebugEnabled())
				log.debug(String.format("Free'd %s task slot %d", phase, next));
			
			/*
			 * When running until a target loss is reached or for a fixed time duration, 
			 * we will not be using a count-down latch but a counter instead.
			 * 
			 * if (latch == null)
			 *		throw new NullPointerException ("error: countdown latch is null");
			 */
			if (latch != null)
				latch.countDown();
			
			numberoftasks.incrementAndGet();
		}
	}
}
