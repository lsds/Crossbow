package uk.ac.imperial.lsds.crossbow.utils;

import java.util.Iterator;

public class CrossbowLinkedList<T> implements Iterable<T> {
	
	/* private static final int _STASH_ = 2; */
	
	class CrossbowListNode {
		
		protected T item;
		protected CrossbowListNode next;
		
		public CrossbowListNode () {
			item = null;
			next = null;
		}
	}
	
	private CrossbowListNode head, tail;
	
	private int size;
	
	/* private CrossbowListNode freeList; */
	
	public CrossbowLinkedList () {
		
		head = tail = null;
		
		size = 0;
		
		/* 
		 * freeList = null;
		 * 
		 * for (int i = 0; i < _STASH_; ++i) 
		 * putNode (new CrossbowListNode ());
		 */
	}
	
	private void putNode (CrossbowListNode node) {
		/*
		 * node.item = null;
		 * node.next = freeList;
		 * freeList = node;
		 */
		return;
	}
	
	private CrossbowListNode getNode () {
		/*
		 * CrossbowListNode node = freeList;
		 * if (node == null)
		 * 	node = new CrossbowListNode ();
		 * else
		 * 	freeList = node.next;
		 * return node;
		 */
		return new CrossbowListNode ();
	}
	
	public int size () {
		return size;
	}
	
	public boolean isEmpty () {
		return (size == 0);
	}
	
	public T peek () {
		if (isEmpty())
			return null;
		return head.item;
	}
	
	public T peek (int order) {
		if (order == 0)
			return peek ();
		/* Double-check that list is not empty */
		if (isEmpty())
			return null;
		CrossbowListNode node = head.next;
		int ord = 1;
		while (node != null) {
			if (ord == order)
				return node.item;
			node = node.next;
			ord ++;
		}
		return null;
	}
	
	public T peekTail () {
		if (isEmpty())
			return null;
		return tail.item;
	}
	
	public void append (T item) {
		CrossbowListNode node = getNode ();
		node.item = item;
		node.next = null;
		if (isEmpty()) {
			head = node;
			tail = node;
		} else {
			tail.next = node;
			tail = node;
		}
		size ++;
		return;
	}
	
	public void prepend (T item) {
		CrossbowListNode node = getNode ();
		node.item = item;
		node.next = null;
		if (isEmpty ()) {
			head = node;
			tail = node;
		} else {
			node.next = head;
			head = node;
		}
		size ++;
		return;
	}
	
	public T removeFirst () {
		CrossbowListNode node;
		T item;
		if (isEmpty())
			return null;
		node = head;
		item = node.item;
		head = node.next;
		putNode (node);
		if ((--size) == 0)
			tail = null;
		return item;
	}
	
	public Iterator<T> iterator () {
		return new Itr ();
	}
	
	private class Itr implements Iterator<T> {
		
		private CrossbowListNode sentinel;
		
		public Itr () {
			sentinel = head;
		}
		
		public boolean hasNext () {
			return (sentinel != null);
		}
		
		public T next () {
			T item = sentinel.item;
			sentinel = sentinel.next;
			return item;
		}
		
		public void remove () {
			throw new IllegalStateException ();
		}
	}
}
