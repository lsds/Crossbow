package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import java.util.concurrent.atomic.AtomicMarkableReference;

import uk.ac.imperial.lsds.crossbow.utils.Queued;

public class LockFreeTaskQueueImpl<T extends Queued> implements ITaskQueue<T> {
	
	class TaskNode {
		
		int key;
		T item;
		
		public AtomicMarkableReference<TaskNode> next;
		
		TaskNode (T item) {
			this.item = item;
			key = item.getKey();
		}
		
		TaskNode (int key, T item) {
			this.key = key;
			this.item = item;
			
			next = new AtomicMarkableReference<TaskNode>(null, false);
		}
	}
	
	class TaskWindow {
		
		TaskNode pred, curr;
		
		public TaskWindow (TaskNode pred, TaskNode curr) {
			this.pred = pred;
			this.curr = curr;
		}
	}
	
	private TaskNode head, tail;
	
	public LockFreeTaskQueueImpl () {
		
		head = new TaskNode (Integer.MIN_VALUE, null);
		tail = new TaskNode (Integer.MAX_VALUE, null);
		
		while (! head.next.compareAndSet(null, tail, false, false));
	}
	
	private TaskWindow find (int key) {
		
		TaskNode pred = null;
		TaskNode curr = null;
		TaskNode succ = null;
		
		boolean [] marked = { false };
		boolean snip;
		
		retry: while (true) {
			
			pred = head;
			curr = pred.next.getReference();
			
			while (true) {
				
				succ = curr.next.get(marked);
				
				while (marked[0]) {
					
					snip = pred.next.compareAndSet(curr, succ, false, false);
					if (! snip)
						continue retry;
					
					curr = pred.next.getReference();
					succ = curr.next.get(marked);
				}
				
				if ((curr.key >= key))
					return new TaskWindow (pred, curr);
				
				pred = curr;
				curr = succ;
			}
		}
	}
	
	public int size () {
		
		return 0;
	}

	public boolean add (T item) {
		
		while (true) {
			
			TaskWindow window = find (tail.key);
			
			TaskNode pred = window.pred;
			TaskNode curr = window.curr;
			
			if (curr.key != tail.key)
				return false;
			else {
				
				TaskNode node = new TaskNode (item);
				node.next = new AtomicMarkableReference<TaskNode>(curr, false);
				
				if (pred.next.compareAndSet(curr, node, false, false)) {
					return true; 
				}
			}
		}
	}
	
	public T poll () {
		
		boolean snip;
		
		while (true) {
			
			TaskWindow window = find (head.key);
			
			TaskNode pred = window.pred;
			TaskNode curr = window.curr;
			
			/* Check if `curr` is not the tail of the queue */
			if (curr.key == tail.key) {
				return null;
			} else {
				/* Mark `curr` as logically removed */
				TaskNode succ = curr.next.getReference();
				snip = curr.next.compareAndSet(succ, succ, false, true);
				if (!snip)
					continue;
				pred.next.compareAndSet(curr, succ, false, false); 
				/* Nodes are rewired */
				return curr.item;
			}
		}
	}
}
