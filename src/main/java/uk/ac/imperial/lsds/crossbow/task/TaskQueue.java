package uk.ac.imperial.lsds.crossbow.task;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.types.SchedulingPolicy;

/*
 * Based on the non-blocking queue of M. Herlihy and N. Shavit
 * from "The Art of Multiprocessor programming".
 */
public class TaskQueue {
	
	private SchedulingPolicy policy;
	
	private AbstractTask head;
	
	public TaskQueue () {
		policy = SystemConf.getInstance().getSchedulingPolicy();
		head = new Task ();
		AbstractTask tail = new Task (Integer.MAX_VALUE, null, null, null, null);
		while (! head.next.compareAndSet(null, tail, false, false));
	}
	
	/*
	 * Inserts task at the end of the queue (lock-free)
	 */
	public boolean add (AbstractTask task) {
		while (true) {
			TaskWindow window = TaskWindow.findTail (head);
			AbstractTask pred = window.pred;
			AbstractTask curr = window.curr;
			if (curr.taskId != Integer.MAX_VALUE) {
				return false;
			} else {
				task.next.set(curr, false);
				if (pred.next.compareAndSet(curr, task, false, false)) {
					return true; 
				}
			}
		}
	}
	
	private AbstractTask getNextTask (int [][] matrix, int p, Integer replicaId, int clock) {
		boolean snip;
		while (true) {
			
			TaskWindow window;
			if (policy == SchedulingPolicy.HLS)
				window = TaskWindow.findNextSkipCost(head, matrix, p, replicaId, clock);
			else
				window = TaskWindow.find (head, replicaId, clock);
			
			AbstractTask pred = window.pred;
			AbstractTask curr = window.curr;
			
			/* Check if `curr` is not the tail of the queue */
			if (curr.taskId == Integer.MAX_VALUE) {
				return null;
			} else {
				/* Mark `curr` as logically removed */
				AbstractTask succ = curr.next.getReference();
				snip = curr.next.compareAndSet(succ, succ, false, true);
				if (!snip)
					continue;
				pred.next.compareAndSet(curr, succ, false, false); 
				/* Nodes are rewired */
				return curr;
			}
		}
	}
	
	public AbstractTask poll (int [][] matrix, int p, Integer replicaId, int clock) {
		if (policy == SchedulingPolicy.NULL)
			return null;
		return getNextTask(matrix, p, replicaId, clock);
	}
	
	/* 
	 * Wait-free, but approximate queue size 
	 */
	public int size () {
		boolean [] marked = { false };
		int count = 0;
		AbstractTask t;
		for (t = head.next.getReference(); t != null && !marked[0]; t = t.next.get(marked)) {
			if (t.taskId < Integer.MAX_VALUE) {
				count ++;
			}
		}
		return count;
	}
	
	/* 
	 * Wait-free, but approximate print out (for debugging) 
	 */
	public void dump (int limit) {
		boolean [] marked = { false };
		int count = 0;
		System.out.print("Q: ");
		AbstractTask t;
		for (t = head.next.getReference(); t != null && !marked[0]; t = t.next.get(marked)) {
			if (t.taskId < Integer.MAX_VALUE) {
		 		System.out.print(String.format("%s ", t.toString()));
				count ++;
			}
			if (count > limit)
				break;
		}
		System.out.println(String.format("(%d tasks)", count));
	}
}
