package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import uk.ac.imperial.lsds.crossbow.utils.Queued;

public class LockBasedTaskQueueImpl<T extends Queued> implements ITaskQueue<T> {
	
	class TaskNode {
		
		int key;
		T item;
		
		TaskNode next;
		
		TaskNode (int key, T item) {
			this.item = item;
			this.next = null;
		}
	}
	
	Lock mutex;
	
	TaskNode head, tail;
	
	TaskNode freeList;
	
	int size;
	
	public LockBasedTaskQueueImpl (int poolSize) {
		
		mutex = new ReentrantLock ();
		
		head = null;
		tail = null;
		
		freeList = null;
		
		size = 0;
		
		for (int i = 0; i < poolSize; i++) {
			putNode (new TaskNode(Integer.MIN_VALUE, null));
		}
	}
	
	/* Return node to free list */
	private void putNode (TaskNode node) {
		
		node.key = Integer.MIN_VALUE;
		node.item = null;
		
		node.next = freeList;
		freeList = node;
		
		return;
	}
	
	private TaskNode getNode (T item) {
		
		TaskNode node = null;
		
		/* If `freeList` is null, all nodes are in use. */
		if ((node = freeList) == null)
			return new TaskNode(item.getKey(), item);
		
		node.key = item.getKey();
		node.item = item;
		
		freeList = node.next;
		
		return node;
	}
	
	public int size () {
		mutex.lock();
		try {
			return size;
		} finally {
			mutex.unlock();
		}
	}
	
	public boolean add (T item) {
		mutex.lock();
		try { 
			/* Thread-safe code */
			TaskNode p = getNode (item);
			p.next = null;
			if (tail != null)
				tail.next = p;
			else
				head = p;
			tail = p;
			size++;
			return true;
		} finally {
			mutex.unlock();
		}
	}
	
	public T poll () {
		T item = null;
		mutex.lock();
		try {
			/* Thread-safe code */
			if (size == 0)
				return null;
			TaskNode node = head;
			item = head.item;
			head = head.next;
			if (head == null)
				tail = null;
			size--;
			putNode(node);
			return item;
		} finally {
			mutex.unlock();
		}
	}
}
