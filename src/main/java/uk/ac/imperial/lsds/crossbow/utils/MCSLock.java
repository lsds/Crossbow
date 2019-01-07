package uk.ac.imperial.lsds.crossbow.utils;

import java.util.concurrent.atomic.AtomicReference;

public class MCSLock {

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

	public MCSLock () {
		
		Q = new AtomicReference<Node>(null);

		node = new ThreadLocal<Node>() {

			protected Node initialValue () {
				return new Node ();
			}
		};
	}

	public void lock () {
		Node p = node.get ();
		Node q = Q.getAndSet (p);
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
			if (Q.compareAndSet(p, null))
				return;
			while (p.next == null) {
				/* Wait until other thread inserts node in queue */
			}
		}
		p.next.locked = false;
		p.next = null;
	}
}
