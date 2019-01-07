package uk.ac.imperial.lsds.crossbow;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;

import uk.ac.imperial.lsds.crossbow.types.Phase;

public class MemoryPlannerV2 {

	private static class Reference {
		
		DataflowNode node;
		int id; /* The node's operator id */
		int referenceCount;

		public Reference (DataflowNode node, int id, int referenceCount) {
			this.node = node;
			this.id = id;
			this.referenceCount = referenceCount;
		}

		public Reference () {
			this(null, 0, 0);
		}

		public Reference (Reference copy) {
			this(copy.node, copy.id, copy.referenceCount);
		}
	}

	private static HashMap<Integer, LinkedList<Reference>> copy (HashMap<Integer, LinkedList<Reference>> map) {
		
		HashMap<Integer, LinkedList<Reference>> map_ = new HashMap<Integer, LinkedList<Reference>> ();
		
		for (Map.Entry<Integer, LinkedList<Reference>> entry : map.entrySet()) {
			
			Integer key = entry.getKey();
			LinkedList<Reference> value  = entry.getValue();
			
			/* Create a deep copy of the value as well */
			LinkedList<Reference> value_ = new LinkedList<Reference> ();
			for (Reference ref: value)
				value_.add(new Reference(ref));
			
			map_.put(key, value_);
		}
		return map_;
	}
	
	/* 
	 * Find list of buffers whose capacity is greater or equal 
	 * to the requested capacity
	 */
	private static LinkedList<Integer> getValidKeys (HashMap<Integer, LinkedList<Reference>> buffers, int capacity) {
		
		LinkedList<Integer> keys = new LinkedList<Integer>();
		
		Iterator<Integer> iterator = buffers.keySet().iterator();
		while (iterator.hasNext()) {
			
			Integer key = iterator.next();
			
			if (key.intValue() >= capacity)
				keys.add(key);
		}
		
		return keys;
	}

	/* Find a reference for a particular operator
	 */
	private static Reference findOperator (HashMap<Integer, LinkedList<Reference>> buffers, int id) {
		
		Iterator<Integer> iterator = buffers.keySet().iterator();
		
		while (iterator.hasNext ()) {
			
			Integer key = iterator.next();
			LinkedList<Reference> refs = buffers.get(key);
			for (Reference ref: refs) {
				if (ref.id == id)
					return ref;
			}
		}
		
		/* Reaching this point in the code indicates an error */
		throw new NullPointerException ();
	}

	private static Reference findUnusedBuffer (HashMap<Integer, LinkedList<Reference>> buffers, int capacity) {
		
		LinkedList<Integer> keys = getValidKeys (buffers, capacity);
		
		if (! keys.isEmpty()) {
			
			for (Integer key: keys) {
				
				LinkedList<Reference> refs = buffers.get(key);
				for (Reference ref: refs) {
					if (ref.referenceCount == 0)
						return ref;
				}
			}
		}
		
		return null;
	}

	private static int capacity (int elements) {
		/* Work with elements (x4 for bytes) */
		return elements;
	}

	public static DataflowNode [] analyse (SubGraph graph) {

		HashMap<Integer, LinkedList<Reference>> buffers = new HashMap<Integer, LinkedList<Reference>> ();
		
		/* A map indicating that op `key` uses a buffer allocated for op `value` */
		HashMap<Integer, Integer> ops = new HashMap<Integer, Integer>();
		
		create (graph, buffers, ops);
		
		/* Copy contents of `buffers` */
		HashMap<Integer, LinkedList<Reference>> buffers_ = copy (buffers);

		DataflowNode [] plan = reuse (graph, buffers, buffers_, ops);

		return plan;
	}

	private static void create (SubGraph graph, 
		
		HashMap<Integer, LinkedList<Reference>> buffers, HashMap<Integer, Integer> ops) {

		/* Traverse subgraph's topology */
		DataflowNode next = graph.getDataflowNode ();
		while (next != null) {

			/* Deal with input buffer(s) */

			Operator [] upstreams = next.getPreviousOperators ();
			if (upstreams != null) {
				
				for (Operator op : upstreams) {
					
					if (! next.getOperator().getKernel().allowsInputOverwrite()) {
						
						int owner = ops.get (op.getId());
						/* Assumes that there is only one output per operator */
						Reference ref = findOperator (buffers, owner);
						ref.referenceCount ++;
					}
				}
			}
			
			/* Deal with output buffer */

			int capacity = capacity(next.getOperator().getOutputShape().countAllElements());
			/* Create a new reference */
			Reference ref = new Reference ();
			ref.node = next;
			ref.id = next.getOperator().getId();
			if (next.getOperator().getKernel().isAccuracyKernel()) {
				ref.referenceCount = (next.getOperator().getKernel().allowsOutputOverwrite() ? 0 : 1);
			} else {
				ref.referenceCount = (next.getOperator().getKernel().allowsOutputOverwrite() ? 0 : next.getOperator().getPeerReferences());
			}
			
			LinkedList<Reference> refs = buffers.get (capacity);
			if (refs == null)
				refs = new LinkedList<>();
			refs.add (ref);
			buffers.put (capacity, refs);

			/* Remember that this operator uses its own output buffer pool */
			ops.put (next.getOperator().getId(), next.getOperator().getId());
			
			next = next.getNextInTopology();
		}

		return;
	}

	private static DataflowNode [] reuse (SubGraph graph, 
		
		HashMap<Integer, LinkedList<Reference>> buffers, 
		HashMap<Integer, LinkedList<Reference>> buffers_, 
		HashMap<Integer, Integer> ops) {

		DataflowNode [] plan = new DataflowNode [graph.numberOfOperators()];
		Arrays.fill(plan, null);
		/* 
		 * A value of -1 indicates a new output buffer pool.
		 * A value of  0 or greater indicates that the i-th 
		 * node can reuse a previously allocated buffer.
		 * 
		 * The `plan` is indexed by (topological) order and 
		 * its values refer to topological order too. 
		 */

		Phase phase = graph.getDataflow().getPhase ();

		/* Traverse subgraph's topology */
		DataflowNode next = graph.getDataflowNode ();
		while (next != null) {

			int capacity = capacity(next.getOperator().getOutputShape().countAllElements());

			/* Look for unused buffers in `buffers` */
			Reference unused;

			if ((next.getOperator().getKernel().isAccuracyKernel()) || (capacity <= 1))
				unused = null;
			else {
				unused = findUnusedBuffer (buffers, capacity);
				if (unused != null) {
					if (next.getOperator().isGradient ()) {
						Operator peer = next.getOperator().getPeer();
						if (peer == unused.node.getOperator())
							unused = null;
					}
				}
			}
		
			if (unused != null) {
				/* Reuse buffer from a previous operator */
				if (unused.referenceCount != 0)
					throw new IllegalStateException ();

				/* Increment the counter if overwriting the output buffer is not allowed */
				Reference ref = findOperator (buffers_, next.getOperator().getId());
				unused.referenceCount = ref.referenceCount;

				plan [next.getOrder()] = unused.node;

				/* Remember that this operator uses the buffer from another operator as its output buffer */
				ops.put (next.getOperator().getId(), unused.id);
			}
			
			/* Decrement reference counters */
			
			if (next.getOperator().isGradient ()) {

				int owner;

				/* Step 1: Decrement the counter of peer's output buffer */

				Operator peer = next.getOperator().getPeer();
				/* Find the owner of the peer's output buffer */
				owner = ops.get (peer.getId());
				/* Find the buffer allocated for that owner */
				Reference ref = findOperator (buffers, owner);

				if (! peer.getKernel().allowsOutputOverwrite ()) {
					if (ref.referenceCount > 0)
						ref.referenceCount --;
				}

				/* Step 2: Decrement the counter of peer's input buffer(s), if needed */

				Operator [] peerUpstreams = peer.getDataflowNode(phase).getPreviousOperators();
				if (peerUpstreams != null) {
					for (Operator op : peerUpstreams) {
						owner = ops.get (op.getId());
						ref = findOperator (buffers, owner);
						if (! peer.getKernel().allowsInputOverwrite()) {
							if (ref.referenceCount > 0)
								ref.referenceCount--;
						}
					}
				}
			}

			/* Decrement the counter(s) of upstream nodes if they are gradients */
			/*
			Operator [] upstreams = next.getPreviousOperators ();
			if (upstreams != null) {
				for (Operator op : upstreams) {
					if (op.isGradient()) {
						int owner = ops.get (op.getId());
						Reference ref = findOperator (buffers, owner);
						if (ref.referenceCount > 0)
							ref.referenceCount --;
					}
				}
			}
			*/

			next = next.getNextInTopology();
		}

		return plan;
	}
}
