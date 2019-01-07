package uk.ac.imperial.lsds.crossbow;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class PerformanceMonitorQueue {
	
	private final static Logger log = LogManager.getLogger (PerformanceMonitorQueue.class);
	
	private static class Node {
		
		long id;
		double throughput;
		Node next, prev;
		
		public Node () {
			clear ();
		}
		
		public void clear () {
			id = 0L;
			throughput = 0D;
			next = prev = null;
		}
		
		public void set (long id, double throughput) {
			this.id = id;
			this.throughput = throughput;
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
	
	Node head, tail;
	
	int size;
	int limit;
	
	boolean bounded;
	
	NodeFactory factory;
	NodeIterator iterator;
	
	Lock lock;
	
	long autoincrement;
	long last;
	
	public PerformanceMonitorQueue (int max, boolean bounded) {
		
		head = new Node (); 
		tail = new Node ();
		head.next = tail;
		tail.prev = head;
		size = 0;
		
		limit = max;
		factory = new NodeFactory (limit);
		this.bounded = bounded;
		
		log.debug(String.format("Initialise performance monitor queue (%d nodes pooled, %sbounded)", limit, ((! bounded) ? "un" : "")));
		
		iterator = new NodeIterator ();
		lock = new ReentrantLock ();
		
		autoincrement = 0L;
		last = 0L;
	}
	
	public boolean isEmpty () {
		return (size == 0);
	}
	
	public int size () {
		return size;
	}
	
	public void add (double throughput) {
		Node p = factory.get();
		p.set (++autoincrement, throughput);
		lock.lock();
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
		factory.free(p);
	}
	
	public double getLastThroughput () {
		double throughput = 0L;
		lock.lock();
		if (! isEmpty ()) {
			if (last < tail.prev.id)
				throughput = tail.prev.throughput;
			last = tail.prev.id;
		}
		lock.unlock();
		return throughput;
	}
	
	public void dump () {
		/* Iterate over the performance monitor queue and dump results */
		lock.lock();
		System.out.println(String.format("=== [Performance monitor queue: %d measurements] ===", size));
		iterator.reset();
		while (iterator.hasNext()) {
			Node p = iterator.next();
			System.out.println(String.format("%9.3f MB/s", p.throughput));
		}
		System.out.println("=== [End of performance monitor measurements dump] ===");
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
}
