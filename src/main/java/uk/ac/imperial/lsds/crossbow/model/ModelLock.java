package uk.ac.imperial.lsds.crossbow.model;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class ModelLock {
	
	private static class Node {
		
		volatile boolean locked;
		Node next;
		
		public Node () {
			
			locked = false;
			next = null;
		}
	}
	
	private AtomicReference<Node> Q;
	private ThreadLocal<Node> node;
	
	public AtomicInteger readers;
	
	public ModelLock () {
		
		Q = new AtomicReference<Node>(null);
		
		node = new ThreadLocal<Node>() {
			
			protected Node initialValue () {
				return new Node ();
			}
		};
		
		readers = new AtomicInteger(0);
	}
	
	/* The method blocks until all writers have finished
	 * (the model is write-locked when readers < 0).
	 */
	public int incReaders () {
		int val = readers.get();
		while ((val < 0) || (! readers.compareAndSet(val, val + 1)))
			val = readers.get();
		return val + 1;
	}
	
	/* Eventually the counter reaches 0, at which point 
	 * the model becomes available to writers.
	 *
	 * The counter keeps decreasing based on the number
	 * of writers.
	 */
	public int decReaders () {
		int val = readers.get();
		while (! readers.compareAndSet(val, val - 1))
			val = readers.get();
		return val - 1;
	}
	
	public int decWriters () {
		int val = readers.get();
		while (! readers.compareAndSet(val, val + 1))
			val = readers.get();
		return val + 1;
	}
	
	public int incWriters () {
		return decReaders();
	}
	
	public boolean isWriteLocked () {
		return (readers.get() < 0);
	}
	
	/* Assume that the counter is 0 (no readers) and try to set 
	 * it to -1. Note that multiple writers can be busy waiting 
	 * but only one will succeed.
	 *
	 * The writer that is successful decrements the counter.
	 */
	private boolean writeLock () {
		while (true) {
			if (isWriteLocked ())
				return false;
			else if (readers.compareAndSet(0, -1))
				return true;
		}
	}
	
	private boolean tryWriteLock() {
		return readers.compareAndSet(0, -1);
	}
	
	public boolean tryLock () {
		
		if (! tryWriteLock())
			return false;
		
		Node p = node.get();
		if (! Q.compareAndSet(null, p)) { 
			/* 
			 * The thread is not at the head of the queue, 
			 * so exit gracefully 
			 */
			decWriters();
			return false;
		}
		
		return true;
	}
	
	public void lock () {
		
		if (! writeLock())
			incWriters();
		
		/* Once the model is write-locked,
		 * multiple writers queue up.
		 */ 
		Node p = node.get();
		Node q = Q.getAndSet(p);
		if (q != null) {
			p.locked = true;
			q.next = p;
			while (p.locked) {
				/* Spin */
			}
		}
	}
	
	public void unlock () {
		Node p = node.get();
		if (p.next == null) {
			if (Q.compareAndSet(p, null)) {
				decWriters();
				return;
			}
			while (p.next == null) {
				/* Wait until other thread inserts node in queue */
			}
		}
		p.next.locked = false;
		p.next = null;
		decWriters();
	}
}
