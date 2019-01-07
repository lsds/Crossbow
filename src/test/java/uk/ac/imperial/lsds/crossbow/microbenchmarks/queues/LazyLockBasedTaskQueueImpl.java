package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import uk.ac.imperial.lsds.crossbow.utils.Queued;

public class LazyLockBasedTaskQueueImpl<T extends Queued> implements ITaskQueue<T> {
	
	class TaskNode {
		
		int key;
		T item;
		
		TaskNode next;
		
		boolean mark;
		Lock mutex;
		
		TaskNode (int key, T item) {
			
			this.key = key;
			this.item = item;
			
			next = null;
			
			mark = false;
			mutex = new ReentrantLock();
		}
		
		public void lock () {
			mutex.lock();
		}
		
		public void unlock () {
			mutex.unlock();
		}
	}
	
	TaskNode head, tail;
	
	public LazyLockBasedTaskQueueImpl () {
		
		head = new TaskNode (Integer.MIN_VALUE, null);
		tail = new TaskNode (Integer.MAX_VALUE, null);
		
		head.next = tail;
	}
	
	public int size () {
		
		int count = 0;
		
		TaskNode curr = head;
		
		while (curr != tail) {
			
			curr = curr.next;
			if (! curr.mark)
				count++;
		}
		
		return count;
	}
	
	private boolean validate (TaskNode pred, TaskNode curr) {
		
		return  (! pred.mark) && (! curr.mark) && (pred.next == curr);
	}
	
	public boolean add (T item) {
		
		while (true) {
			
			TaskNode pred = head;
			TaskNode curr = head.next;
			
			/* Find tail */
			while (curr.key < Integer.MAX_VALUE) {
				pred = curr;
				curr = curr.next;
			}
			
			pred.lock();
			try {
				curr.lock();
				try {	
					if (validate (pred, curr)) {	
						TaskNode node = new TaskNode (item.getKey(), item);
						node.next = curr;
						pred.next = node;
						return true;
					}
				} finally {
					curr.unlock();
				}
			} finally {
				pred.unlock();
			}
		}
	}
	
	public T poll () {
		
		while (true) {
			
			TaskNode pred = head;
			TaskNode curr = head.next;
			
			/**
			 * Iterate over tasks until condition is met
			 *
			 * while (condition) {	
			 *	pred = curr; 
			 * 	curr = curr.next;
			 * }
			 */
			
			pred.lock();
			try {
				curr.lock();
				try {	
					if (validate (pred, curr)) {	
						/* Is the queue empty? */
						if (curr == tail) {
							return null;
						} else {
							T t = curr.item;
							curr.mark = true;
							pred.next = curr.next;
							return t;
						}
					}
				} finally {
					curr.unlock();
				}
			} finally {
				pred.unlock();
			}
		}
	}
}
