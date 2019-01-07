package uk.ac.imperial.lsds.crossbow.utils;

import java.util.concurrent.atomic.AtomicBoolean;

public class TTASLock {
	
	private AtomicBoolean state;
	
	public TTASLock () {
		state = new AtomicBoolean (false);
	}
	
	public void lock () {
		while (true) {
			while (state.get()) {};
			if (! state.getAndSet(true))
				return;
		}
	}
	
	public boolean tryLock () {
		if (! state.getAndSet(true)) {
			return true;
		} 
		else {
			return false;
		}
	}
	
	public void unlock () {
		state.set(false);
	}
}
