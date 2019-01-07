package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class WorkClock {
	
	private final static Logger log = LogManager.getLogger (WorkClock.class);
	
	private int init;
	private int next;
	private int innerclock;
	private int clock;
	private int wraps;
	
	public int wpc;
	
	private int max;
	
	private int getAdaptiveWpc () {
		return wpc;
	}
	
	public WorkClock (int init, int wpc, int max) {
		
		this.init = init;
		this.next = init;
		this.innerclock = 0;
		this.clock = 1;
		this.wraps = 0;
		
		this.wpc = wpc;
		
		this.max = max;
		
		if (this.max > wpc)
			while (((this.max - 1) % wpc) != 0)
				this.max--;
		else
			while ((wpc % (this.max - 1)) != 0)
				this.max--;
		
		log.info(String.format("wpc = %d max = %d", this.wpc, this.max));
	}
	
	public boolean isBarrier (int next) {
		
		if (this.next != next)
			throw new IllegalStateException("error: illegal state in work clock");
		
		return ((innerclock + 1) > getAdaptiveWpc());
	}
	
	public int getNext () {
		return next;
	}
	
	public int getClock () {
		return clock;
	}
	
	public int numberOfWraps () {
		return wraps;
	}
	
	private void incrementNext () {
		
		if (++next == max) {
			next = init + 1;
			wraps ++;
		}
		if (++innerclock > getAdaptiveWpc()) {
			innerclock = 1;
			clock ++;
		}
		return;
	}
	
	public int incrementAndGetNext (int [] currentClock) {
		incrementNext ();
		if (currentClock != null)
			currentClock[0] = clock;
		return next;
	}
	
	public int getMax () {
		return max;
	}
	
	public String info () {
		StringBuilder s = new StringBuilder("[");
		s.append(String.format("next %4d clock %4d:%2d %2d wraps", next, clock, innerclock, wraps));
		s.append("]");
		return s.toString();
	}
}
