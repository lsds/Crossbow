package uk.ac.imperial.lsds.crossbow;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

import uk.ac.imperial.lsds.crossbow.types.Phase;

public class MemoryPlanner {

	private static class Reference {

		DataflowNode node;
		int id; /* The node's operator id */

		int referenceCount;
	}

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
		/*
		 * TODO Let's think about sorting order...
		 */
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
		return elements;
	}

	public static DataflowNode [] analyse (SubGraph graph) {

		HashMap<Integer, LinkedList<Reference>> buffers = new HashMap<Integer, LinkedList<Reference>> ();

		DataflowNode [] plan = new DataflowNode [graph.numberOfOperators()];
		/* 
		 * A value of -1 indicates a new output buffer pool.
		 * A value of  0 or greater indicates that the i-th 
		 * node can reuse a previously allocated buffer.
		 * 
		 * The `plan` is indexed by (topological) order and 
		 * its values refer to topological order too. 
		 */
		Arrays.fill(plan, null);

		/* Traverse subgraph's topology */
		DataflowNode next = graph.getDataflowNode ();

		Phase phase = graph.getDataflow().getPhase ();

		HashMap<Integer, Integer> theOtherPlan = new HashMap<Integer, Integer>();

		int requiredCapacity;
		int owner;

		Reference ref;

		while (next != null) {

			System.out.println("[DBG] Find buffer for operator " + next.getOrder());
			
			/* Deal with input buffers */

			Operator [] upstreams = next.getPreviousOperators ();
			
			if (upstreams != null) {

				for (Operator op : upstreams) {
					if (! next.getOperator().getKernel().allowsInputOverwrite()) {

						owner = theOtherPlan.get (op.getId());

						/* Assume that the capacity of the upstream operator and the buffer owner can be used interchangeably */
						ref = findOperator (buffers, owner);

						/* TODO Check if dealing with multiple outputs works... */
						System.out.println("[DBG] Increment ref count for operator " + ref.node.getOrder() + " to " + (ref.referenceCount + 1));
						ref.referenceCount ++;
					}
				}
			}

			/* Deal with output buffer */

			requiredCapacity = capacity(next.getOperator().getOutputShape().countAllElements());
			/* Look for unused buffers in `buffers` */
			Reference unused;
			/* 
			 * TODO Check if dealing with multiple outputs works...
			 * 
			 * The following if statement disables dealing
			 * with multiple outputs.
			 * 
			 * if (next.getNextList() != null && next.getNextList().size() == 1)
			 */
			//            if (next.getNextList() != null && next.getNextList().size() == 1)
			//           unused = findUnusedBuffer (buffers, requiredCapacity);
			//            else
			//              unused = null;

			if (
					// (next.getOperator().getKernel() instanceof ElementWiseOpGradient) || 
					// (next.getOperator().getKernel() instanceof ElementWiseOp) ||
					// (next.getOperator().getName().contains("shortcut")) ||
					// (next.getOperator().getKernel() instanceof ReLUGradient) ||
					// (next.getOperator().getKernel() instanceof BatchNormGradient) ||
					// (next.getOperator().getKernel() instanceof ConvGradient)
					(! next.getOperator().getName().contains("stage-4-unit-1"))
					)
				unused = null;
			else
				unused = findUnusedBuffer (buffers, requiredCapacity);

			if (unused == null) {
				/* Create a new reference */
				ref = new Reference ();

				ref.node = next;
				ref.id = next.getOperator().getId();
				if (next.getOperator().getKernel().isAccuracyKernel()) {
					ref.referenceCount = (next.getOperator().getKernel().allowsOutputOverwrite() ? 0 : 1);
				} else {
					ref.referenceCount = (next.getOperator().getKernel().allowsOutputOverwrite() ? 0 : next.getOperator().getPeerReferences() + ((next.getNextOperators() != null) ? next.getNextOperators().length : 0));
				}
				System.out.println(String.format("[DBG] Created buffer for operator %d with ref. count %d (name is %s)", next.getOrder(), ref.referenceCount, next.getOperator().getName()));

				LinkedList<Reference> refs = buffers.get (requiredCapacity);
				if (refs == null) {
					refs = new LinkedList<>();
				}
				refs.add (ref);
				buffers.put (requiredCapacity, refs);

				/* Remember that this operator uses its own output buffer pool */
				theOtherPlan.put (next.getOperator().getId(), next.getOperator().getId());
			}
			else {
				/* Reuse buffer from a previous operator */
				System.out.println("[DBG] Reusing buffer from operator " + unused.node.getOrder());
				if (unused.referenceCount != 0)
					throw new IllegalStateException ();

				/* Increment the counter if overwriting the output buffer is not allowed */
				unused.referenceCount = (next.getOperator().getKernel().allowsOutputOverwrite() ? 0 : next.getOperator().getPeerReferences()); /* Was 1 */

				plan [next.getOrder()] = unused.node;

				/* Remember that this operator uses the buffer from another operator as its output buffer */
				theOtherPlan.put (next.getOperator().getId(), unused.id);
			}

			/* Decrement counters */

			if (next.getOperator().isGradient ()) {

				/* Step 1: Decrement the counter of peer's output buffer */

				Operator peer = next.getOperator().getPeer();
				/* Find the owner of the peer's output buffer */
				owner = theOtherPlan.get (peer.getId());
				/* Assume that the capacity of peer and the buffer owner can be used interchangeably */
				ref = findOperator (buffers, owner);

				if (! peer.getKernel().allowsOutputOverwrite ()) {
					if (ref.referenceCount > 0) {
						System.out.println("[DBG] Decrement ref count for operator " + ref.node.getOrder() + " to " + (ref.referenceCount - 1));
						ref.referenceCount --;
					}
				}

				/* Step 2: Decrement the counter of peer's input buffer(s), if needed */

				Operator [] peerUpstreams = peer.getDataflowNode(phase).getPreviousOperators();

				if (peerUpstreams != null) {

					for (Operator op : peerUpstreams) {

						owner = theOtherPlan.get (op.getId());
						ref = findOperator (buffers, owner);

						if (! peer.getKernel().allowsInputOverwrite()) {
							if (ref.referenceCount > 0) {
								System.out.println("[DBG] Decrement ref count for operator " + ref.node.getOrder() + " to " + (ref.referenceCount - 1));
								ref.referenceCount--;
							}
						}
					}
				}
			}

			/* Decrement the counter(s) of upstream nodes if they are gradients */

			if (upstreams != null) {
				for (Operator op : upstreams) {
					if (op.isGradient()) {

						owner = theOtherPlan.get (op.getId());
						ref = findOperator (buffers, owner);

						if (ref.referenceCount > 0)
							ref.referenceCount --;
					}
				}
			}

			next = next.getNextInTopology();
		}

		return plan;
	}
}
