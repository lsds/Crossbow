package uk.ac.imperial.lsds.crossbow.scheduler;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

public class DataflowNode {

	public int id;

	public int order;
	public int label;

	public ArrayList<DataflowNode> prev, next;

	public DataflowNode nextInTopology, prevInTopology;

	private boolean marked, visited;

	public DataflowNode (int id) {

		this.id = id;

		order = -1;
		label = -1;

		prev = new ArrayList<DataflowNode>();
		next = new ArrayList<DataflowNode>();

		nextInTopology = prevInTopology = null;

		marked = visited = false;
	}

	public boolean isMostUpstream () {
		return (prev.size() == 0);
	}

	public boolean isMostDownstream () {
		return (next.size() == 0);
	}

	public boolean isDisconnected () {
		return (isMostUpstream () && isMostDownstream ());
	}

	public DataflowNode connectTo (DataflowNode node) {
		next.add(node);
		node.setUpstream(this);
		return node;
	}

	public void setUpstream (DataflowNode node) {
		prev.add(node);
	}

	public DataflowNode setNextInTopology (DataflowNode node) {

		if (this.equals(node)) /* First in topological sort */
			return this;

		nextInTopology = node;
		node.setPreviousInTopology (this);
		return node;
	}

	public void setPreviousInTopology (DataflowNode node) {
		prevInTopology = node;
		return;
	}

	public void visit (LinkedList<DataflowNode> list) {

		if (isMarked ())
			throw new IllegalStateException ("error: graph is not directed acyclic");

		if (! isVisited ()) {
			mark ();
			if (next != null) {
				Iterator<DataflowNode> iterator = next.iterator();
				while (iterator.hasNext()) {
					DataflowNode node = iterator.next();
					node.visit (list);
				}
			}
			setVisited (true);
			unmark ();
			list.addFirst(this);
		}
	}

	private void setVisited (boolean visited) {
		this.visited = visited;
	}

	private boolean isVisited () {
		return visited;
	}

	private DataflowNode mark () {
		this.marked = true;
		return this;
	}

	private DataflowNode unmark () {
		this.marked = false;
		return this;
	}

	private boolean isMarked () {
		return marked;
	}

	public String getLabel() {
		if (label < 0)
			return "x";
		return String.format("%d", label);
	}
}
