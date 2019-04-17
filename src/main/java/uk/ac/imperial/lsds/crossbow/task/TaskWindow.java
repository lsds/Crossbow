package uk.ac.imperial.lsds.crossbow.task;

import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

class TaskWindow {
	
	public AbstractTask pred, curr;
	
	public TaskWindow (AbstractTask pred, AbstractTask curr) {
		this.pred = pred;
		this.curr = curr;
	}
	
	public static TaskWindow find (AbstractTask head, int taskid) {
		AbstractTask pred = null;
		AbstractTask curr = null;
		AbstractTask succ = null;
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
				if ((curr.taskId >= taskid))
					return new TaskWindow (pred, curr);
				pred = curr;
				curr = succ;
			}
		}
	}
	
	public static TaskWindow findTail (AbstractTask head) {
		AbstractTask pred = null;
		AbstractTask curr = null;
		AbstractTask succ = null;
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
				if ((curr.taskId >= Integer.MAX_VALUE))
					return new TaskWindow (pred, curr);
				pred = curr;
				curr = succ;
			}
		}
	}
	
	public static TaskWindow findHead (AbstractTask head) {
		AbstractTask pred = null;
		AbstractTask curr = null;
		AbstractTask succ = null;
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
					curr = succ; // pred.next.getReference();
					succ = curr.next.get(marked);
				}
				return new TaskWindow (pred, curr);
			}
		}
	}
	
	private static boolean select (AbstractTask t, Integer replicaId, int clock) {
		// 
		// There are the following basic options:
		// 
		// 1. worker.replica = null, t.access = NONE, task.replica = null; return  true
		// 
		// Neither the worker nor the task is bound to a model, but the task does not 
		// require any access.
		// 
		// 2. worker.replica = null, t.access = NONE, task.replica = Y; return  true
		// 
		// The worker's replica does not matter; and nor does the task's access type 
		// since a model is already bound to this task's  pipeline.
		// 
		// 3. worker.replica = null, t.access = R/0 or R/W, task.replica = null; return false
		// 
		// The task cannot be executed.
		// 
		// 4. worker.replica = null, t.access = R/0 or R/W, task.replica = Y; return  true
		// 
		// The worker's replica does not matter (like case 2).
		// 
		// 5. worker.replica = X, t.access = NONE, task.replica = null; return true
		// 
		// At this point, we could assign X to be this task's pipeline model replica, 
		// but we can always bind it later (lazy binding). X is released early.
		// 
		// 6. worker.replica = X, t.access = NONE, task.replica = Y; return true
		// 
		// The worker's replica does not matter. X is released early.
		// 
		// 7. worker.replica = X, t.access = R/0 or R/W, task.replica = null; return (worker.wpc >= t.wpc)
		// 
		// X is bound to this task's pipeline, as long as it has a valid wpc counter: 
		// the worker's replica wpc should be greater or equal to the task's wpc.
		// 
		// 8. worker.replica = X, t.access = R/0 or R/W, task.replica = Y; return true
		// 
		// The worker's replica does not matter. X is released early.
		// 
		// System.out.println(String.format("[DBG] select: t %04d t.lowerBound=%d t.replica=%s worker.replica=%s clock=%d", t.taskid, t.lowerBound, t.replicaId, replicaId, clock));
		// 
		if (t.replicaId == null && t.access.compareTo(ModelAccess.NA) > 0) {
			// System.out.println(String.format("[DBG] select: t %04d t.replica=%s worker.replica=%s clock=%d", t.taskid, t.replicaId, replicaId, clock));
			if (replicaId == null) {
				return false;
			} else {
				// System.out.println(String.format("[DBG] select: t %04d t.replica=%s worker.replica=%s clock=%d", t.taskid, t.replicaId, replicaId, clock));
				// Check wpc counter
				if (clock < t.lowerBound) {
					// System.out.println(String.format("[DBG] select failed: t %04d t.lowerBound=%d t.replica=%s worker.replica=%s clock=%d", t.taskid, t.lowerBound, t.replicaId, replicaId, clock));
					return false;
				}
			}
			// Bind model replica to task
			// System.out.println(String.format("[DBG] bind task %d to replica %d", t.taskid, replicaId.intValue()));
			// t.replicaId = replicaId;
		}
		return true;
	}
	
	public static TaskWindow find (AbstractTask head, Integer replicaId, int clock) {
		AbstractTask pred = null;
		AbstractTask curr = null;
		AbstractTask succ = null;
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
				
				if (curr.taskId == Integer.MAX_VALUE)
					return new TaskWindow (pred, curr);
				
				/*
				 * The task's model access requirement should be less that
				 * the worker's granted access, and so should be its WPC. 
				 */
				if (select(curr, replicaId, clock))
					return new TaskWindow (pred, curr);
				
				pred = curr;
				curr = succ;
			}
		}
	}
	
	public static TaskWindow findNextSkipCost (AbstractTask head, int[][] policy, int p, Integer replicaId, int clock) {
		AbstractTask pred = null;
		AbstractTask curr = null;
		AbstractTask succ = null;
		boolean [] marked = { false };
		boolean snip;
		int _p = (p + 1) % 2; /* The other processor */
		double skip_cost = 0.;
		retry: while (true) {
			pred = head;
			curr = pred.next.getReference();
			if (curr.taskId == Integer.MAX_VALUE)
				return new TaskWindow (pred, curr);
			while (true) {
				succ = curr.next.get(marked);
				while (marked[0]) {
					snip = pred.next.compareAndSet(curr, succ, false, false);
					if (! snip)
						continue retry;
					curr = pred.next.getReference();
					succ = curr.next.get(marked);
				}
				
				if (curr.taskId == Integer.MAX_VALUE)
					return new TaskWindow (pred, curr);
				
				if (
					select(curr, replicaId, clock) &&
					(
						policy[p][curr.graphId] >= policy[_p][curr.graphId] || 
						((skip_cost >= 1. / (double) policy[p][curr.graphId]))
					)
				) {
					return new TaskWindow (pred, curr);
				}
				
				skip_cost += 1. / (double) policy[_p][curr.graphId];
				
				pred = curr;
				curr = succ;
			}
		}
	}
}
