package uk.ac.imperial.lsds.crossbow;

import java.util.concurrent.atomic.AtomicBoolean;

public class CountDownTimer implements Runnable {
	
	AtomicBoolean flag;
	long seconds = 0;
	
	public CountDownTimer (AtomicBoolean flag) {
		if (flag.get())
			throw new IllegalStateException ("error: countdown timer flag must be false");
		
		this.flag = flag;
	}
	
	public CountDownTimer set (long seconds) {
		this.seconds = seconds;
		return this;
	}
	
	@Override
	public void run () {
		
		try { 
			
			Thread.sleep (seconds * 1000L);
		
		} catch (Exception e) {

			e.printStackTrace();
			System.exit(1);
		}
		
		flag.set(true);
		return;
	}
}
