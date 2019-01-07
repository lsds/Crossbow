package uk.ac.imperial.lsds.crossbow.scheduler;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;

public class TestLabels {
	
	public static void export (DataflowNode head, String filename) {
		if (filename == null) 
		{
			DataflowNode node = head;
			while (node != null) {
				for (DataflowNode downstream: node.next) {
					System.out.println(String.format("%d (%s) -> %d (%s)\n", node.id, node.getLabel (), downstream.id, downstream.getLabel ()));
				}
				node = node.nextInTopology;
			}
		}
		else 
		{	
			FileWriter f;
			BufferedWriter b;
			DataflowNode node;
			try 
			{	
				System.out.println (String.format("Exporting dataflow graph to %s", filename));
				f = new FileWriter (filename);
				b = new BufferedWriter (f);
				b.write("digraph G {\n");
				node = head;
				b.write("\tnode [shape=circle]\n");
				b.write("\n");
				/* Create nodes */
				while (node != null) {
					b.write(String.format("\tn%d [label=<%d<sup><font color=\"red\"><b>%s</b></font></sup>>]\n", node.id, node.id, node.getLabel()));
					node = node.nextInTopology;
				}
				b.write("\n");
				/* Create edges */
				node = head;
				while (node != null) {
					for (DataflowNode downstream: node.next) {
						b.write(String.format("\tn%d -> n%d\n", node.id, downstream.id));
					}
					node = node.nextInTopology;
				}
				b.write("}\n");
				/* Clean-up */
				b.flush();
				f.flush();
				f.close();
			}
			catch (IOException e) {
				System.err.println("error: failed to export dataflow graph: " + e.getMessage());
				System.exit(1);
			}
		}
	}
	
	public static void topologicalSort (DataflowNode head) {
		
		/* Depth-first topological sort */
		LinkedList<DataflowNode> list = new LinkedList<DataflowNode> ();
		head.visit(list);
		
		/* Configure execution order */
		int order = 0;
		Iterator<DataflowNode> iterator = list.iterator();
		DataflowNode curr = head;
		while (iterator.hasNext()) {
			
			DataflowNode node = iterator.next ();
			curr = curr.setNextInTopology (node);
			
			node.order = order ++;
		}
	}
	
	public static void checkLabel (DataflowNode node) {
		if (node.label < 0) {
			System.err.println("error: label not set");
			System.exit(1);
		}
	}
	
	public static void assignLabels (DataflowNode head) {
		/* Assign label to most upstream node */
		head.label = 1;
		
		int maxlabel = head.label;
		
		DataflowNode node = head;
		while (node != null) {
			checkLabel (node);
			/* Assign labels to downstream nodes, incrementing by 1 on every branch */
			boolean first = true;
			for (DataflowNode downstream: node.next) {
				if (first) {
					downstream.label = node.label;
					first = false;
				}
				else
					downstream.label = ++maxlabel;
			}
			node = node.nextInTopology;
		}
	}
	
	public static void main (String [] args) {
		
		DataflowNode h1 = Examples._1();
		topologicalSort (h1);
		assignLabels (h1);
		export (h1, "/home/akolious/example1.dot");
		
		DataflowNode h2 = Examples._2();
		topologicalSort (h2);
		assignLabels (h2);
		export (h2, "/home/akolious/example2.dot");
		
		DataflowNode h3 = Examples._3();
		topologicalSort (h3);
		assignLabels (h3);
		export (h3, "/home/akolious/example3.dot");
		
		DataflowNode h4 = Examples._4();
		topologicalSort (h4);
		assignLabels (h4);
		export (h4, "/home/akolious/example4.dot");
	}
}
