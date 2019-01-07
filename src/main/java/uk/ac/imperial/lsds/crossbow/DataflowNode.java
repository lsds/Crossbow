package uk.ac.imperial.lsds.crossbow;

import java.util.Iterator;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.utils.CrossbowArrayList;
import uk.ac.imperial.lsds.crossbow.utils.CrossbowLinkedList;

public class DataflowNode {
	
	private final static Logger log = LogManager.getLogger (DataflowNode.class);
	
	/* Execution order in dataflow */
	private int order;
	
	/* Stream label */
	private int label;
	
	/* t-level */
	private int level;
	
	private Operator operator;
	
	private CrossbowArrayList<DataflowNode> prevList;
	private CrossbowArrayList<DataflowNode> nextList;
	
	private Operator [] previousOperators, nextOperators;
	
	private DataflowNode nextInTopology, prevInTopology;
	
	private boolean marked, visited, finalised;
	
	/*
	 * Data re-use
	 * 
	 * We store if and from where we are going to re-use an output buffer.
	 * 
	 * We elevate this logic to the node level because an operator can be 
	 * part of more than one dataflow and we should  permit
	 * different memory plans for each.
	 * 
	 * `donor` is the node whose operator the current node's operator can 
	 * re-use; and `position` refers to the position in the donor's output 
	 * queue.  
	 */
	private DataflowNode donor;
	private int position;
	
	public DataflowNode (Operator op) {
		
		order = -1;
		label = -1;
		level = -1;
		
		operator = op;
		
		prevList = nextList = null;
		
		previousOperators = nextOperators = null;
		
		nextInTopology = prevInTopology = null;
		
		marked = visited = finalised = false;
		
		donor = null;
		position = 0;
	}
	
	public Operator getOperator () {
		
		if (operator == null)
			throw new NullPointerException ("error: dataflow node operator is null");
		
		return operator;
	}
	
	public CrossbowArrayList<DataflowNode> getNextList () {
		return nextList;
	}
	
	public CrossbowArrayList<DataflowNode> getPreviousList () {
		return prevList;
	}
	
	public void setPrevious (DataflowNode node) {
		
		if (prevList == null)
			prevList = new CrossbowArrayList<DataflowNode> ();
		
		prevList.add(node);
		
		return;
	}
	
	public DataflowNode connectTo (DataflowNode node) {
		
		if (nextList == null)
			nextList = new CrossbowArrayList<DataflowNode> ();
		
		nextList.add(node);
		node.setPrevious (this);
		
		return node;
	}
	
	public void setPreviousInTopology (DataflowNode node) {
		prevInTopology = node;
		return;
	}
	
	public DataflowNode getPreviousInTopology () {
		return prevInTopology;
	}
	
	public DataflowNode setNextInTopology (DataflowNode node) {
		
		if (this.equals(node)) /* First in topological sort */
			return this;
		
		nextInTopology = node;
		node.setPreviousInTopology (this);
		return node;
	}
	
	public DataflowNode getNextInTopology () {
		return nextInTopology;
	}
	
	public DataflowNode setOrder (int order) {
		this.order = order;
		return this;
	}
	
	public int getOrder () {
		return order;
	}
	
	public DataflowNode setLabel (int label) {
		this.label = label;
		return this;
	}
	
	public int getLabel () {
		return label;
	}
	
	public DataflowNode setLevel (int level) {
		this.level = level;
		return this;
	}
	
	public int getLevel () {
		return level;
	}
	
	public void visit (CrossbowLinkedList<DataflowNode> list) {
		
		log.debug ("Visit " + operator.getName());
		
		if (isMarked ())
			throw new IllegalStateException ("error: sub-graph is not directed acyclic");
		
		if (! isVisited ()) {
			mark ();
			if (nextList != null) {
				Iterator<DataflowNode> iterator = nextList.iterator();
				while (iterator.hasNext()) {
					DataflowNode node = iterator.next();
					node.visit (list);
				}
			}
			setVisited (true);
			unmark ();
			list.prepend (this);
		}
	}
	
	public void setVisited (boolean visited) {
		this.visited = visited;
	}
	
	public boolean isVisited () {
		return visited;
	}
	
	public DataflowNode mark () {
		this.marked = true;
		return this;
	}
	
	public DataflowNode unmark () {
		this.marked = false;
		return this;
	}
	
	public boolean isMarked () {
		return marked;
	}
	
	public DataflowNode finalise () {
		
		if (prevList != null) {
			int size = prevList.size();
			previousOperators = new Operator [size];
			for (int i = 0; i < size; ++i)
				previousOperators [i] = prevList.get(i).getOperator();
		}
		
		if (nextList != null) {
			int size = nextList.size();
			nextOperators = new Operator [size];
			for (int i = 0; i < size; ++i)
				nextOperators [i] = nextList.get(i).getOperator();
		}
		
		setFinalised (true);
		return this;
	}
	
	public void setFinalised (boolean finalised) {
		this.finalised = finalised;
	}
	
	public boolean isFinalised () {
		return finalised;
	}
	
	public DataflowNode shallowCopy () {
		return new DataflowNode (operator);
	}

	public Operator [] getPreviousOperators () {
		return previousOperators;
	}
	
	public Operator [] getNextOperators () {
		return nextOperators;
	}
	
	public DataflowNode getOutputBufferFrom (DataflowNode provider) {
		this.donor = provider;
		return this;
	}
	
	public DataflowNode getOutputBufferDonor () {
		return donor;
	}
	
	public boolean getOutputBufferFromElsewhere () {
		return (donor != null);
	}
	
	public DataflowNode getOutputBufferFromPosition (int position) {
		this.position = position;
		return this;
	}
	
	public int getOutputBufferFromPosition () {
		return position;
	}
	
	public String exportDot () {
		StringBuilder s = new StringBuilder (String.format("\tn%d [shape=plaintext label=<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\" cellpadding=\"10\">", getOrder()));
		s.append(String.format("<tr><td><b>%d</b></td><td><font color=\"red\"><b>%d</b></font></td><td>%s</td></tr>", getOrder(), getLabel(), getOperator().getName()));
		s.append("</table>>];\n");
		return s.toString();
	}
}
