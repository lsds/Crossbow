package uk.ac.imperial.lsds.crossbow.result;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.types.Phase;

public class MeasurementQueue {
	
	private final static Logger log = LogManager.getLogger (MeasurementQueue.class);
	
	private static class Node {
		
		long timestamp;
		float loss, accuracy;
		Node next, prev;
		
		public Node () {
			clear ();
		}
		
		public void clear () {
			timestamp = 0L;
			loss = accuracy = 0F;
			next = prev = null;
		}
		
		public void set (long timestamp, float loss, float accuracy) {
			this.timestamp = timestamp;
			this.loss = loss;
			this.accuracy = accuracy;
		}
	}
	
	private static class NodeFactory {
		
		private int poolSize;
		
		private ConcurrentLinkedQueue<Node> pool = new ConcurrentLinkedQueue<Node>();
		
		public NodeFactory (int size) {
			poolSize = size;
			int i = poolSize;
			while (i-- > 0)
				pool.add(new Node());
		}
		
		public Node get () {
			Node p = pool.poll();
			if (p == null)
				return new Node ();
			return p;
		}
		
		public void free (Node p) {
			p.clear();
			pool.offer (p);
		}
	}
	
	private class NodeIterator {
		
		private Node current, last;
		private int index;
		
		public NodeIterator () {
			reset();
		}
		
		public void reset () {
			current = head.next;
			last = null;
			index = 0;
		}
		
		public boolean hasNext () {
			return (index < size);
		}
		
		public Node next () {
			Node result;
			if (! hasNext())
				throw new IllegalStateException ("error: invalid pointer access in node iterator");
			last = current;
			result = current;
			current = current.next;
			++index;
			return result;
		}
		
		public void remove () {
			if (last == null)
				throw new IllegalStateException ("error: invalid pointer access in node iterator");
			Node f = last;
			Node q = last.prev;
			Node p = last.next;
			q.next = p;
			p.prev = q;
			--size;
			if (current == last)
				current = p;
			else
				--index;
			last = null;
			factory.free(f);
		}
	}
	
	Phase phase;
	
	Node head, tail;
	
	int size;
	int limit;
	
	boolean bounded;
	
	NodeFactory factory;
	NodeIterator iterator;
	
	float accumulatedLoss;
	
	float accumulatedAccuracy;
	
	Lock lock;
	
	public MeasurementQueue (Phase phase, int max, boolean bounded) {
		
		if (max < 2) {
			/* throw new IllegalArgumentException ("error: measurement queue limit must be greater than 1"); */
		}
		
		this.phase = phase;
		
		head = new Node (); 
		tail = new Node ();
		head.next = tail;
		tail.prev = head;
		size = 0;
		
		limit = max;
		factory = new NodeFactory (limit);
		this.bounded = bounded;
		
		accumulatedLoss = 0;
		accumulatedAccuracy = 0;
		
		log.debug(String.format("Initialise measurement queue (%d nodes pooled, %sbounded) for %s phase", limit, ((! bounded) ? "un" : ""), phase.toString()));
		
		iterator = new NodeIterator ();
		lock = new ReentrantLock ();
	}
	
	public boolean isEmpty () {
		return (size == 0);
	}
	
	public void add (long timestamp, float loss, float accuracy) {
		Node p = factory.get();
		p.set (timestamp, loss, accuracy);
		lock.lock();
		accumulatedLoss += loss;
		accumulatedAccuracy += accuracy;
		Node last = tail.prev;
		p.next = tail;
		p.prev = last;
		tail.prev = p;
		last.next = p;
		++size;
		if (size >= limit && bounded)
			remove ();
		lock.unlock();
	}
	
	public void remove () {
		Node p = head.next;
		Node q = p.next;
		head.next = q;
		q.prev = head;
		--size;
		accumulatedLoss -= p.loss;
		accumulatedAccuracy -= p.accuracy;
		factory.free(p);
	}
	
	public float getAccumulatedLoss () {
		return accumulatedLoss;
	}
	
	public float getAccumulatedAverageLoss () {
		if (size == 0)
			return 0;
		return (accumulatedLoss / (float) size);
	}
	
	public float getAccumulatedAccuracy () {
		return accumulatedAccuracy;
	}
	
	public float getAccumulatedAverageAccuracy () {
		if (size == 0)
			return 0;
		return (accumulatedAccuracy / (float) size);
	}
	
	public long getFirstTimestamp () {
		if (isEmpty())
			return -1L;
		return head.next.timestamp;
		
	}
	
	public long getLastTimestamp () {
		if (isEmpty())
			return -1L;
		return tail.prev.timestamp;
	}
	
	public void dump () {
		/* Iterate over the measurements queue and dump results */
		lock.lock();
		/* Configure start and end timestamps */
		long start = getFirstTimestamp();
		long end = getLastTimestamp();
		System.out.println(String.format("=== [%s loss & accuracy: %d measurements, %5.5f secs] ===", phase.toString(), size, ((double) (end - start) / 1000000000D)));
		iterator.reset();
		while (iterator.hasNext()) {
			Node p = iterator.next();
			System.out.println(String.format("%13d %5.5f %5.5f", p.timestamp - start, p.loss, p.accuracy));
		}
		System.out.println("=== [End of measurements dump] ===");
		lock.unlock();
		return;
	}
	
	public void releaseAll () {
		/* Iterate over the measurements queue and remove nodes (returned to the queue) */
		lock.lock();
		iterator.reset();
		while (iterator.hasNext())
			iterator.remove();
		lock.unlock();
		return;
	}

	public void check (float threshold, int occurencies) {
		int count = 0;
		lock.lock();
		iterator.reset();
		while (iterator.hasNext()) {
			Node p = iterator.next();
			if (p.accuracy < threshold)
				count ++;
			else
				count --;
		}
		if (count > occurencies) {
			System.err.println("More than 10 measurements with accuracy less than 0.1. Exiting assuming divergence.");
			System.exit(1);
		}	
		lock.unlock();
	}
}
