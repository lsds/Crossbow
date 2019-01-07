package uk.ac.imperial.lsds.crossbow;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.dispatcher.ITaskDispatcher;
import uk.ac.imperial.lsds.crossbow.dispatcher.LightWeightTaskDispatcher;
import uk.ac.imperial.lsds.crossbow.dispatcher.ResNet50TaskDispatcher;
import uk.ac.imperial.lsds.crossbow.dispatcher.TaskDispatcher;
import uk.ac.imperial.lsds.crossbow.result.IResultHandler;
import uk.ac.imperial.lsds.crossbow.result.ResultCollector;
import uk.ac.imperial.lsds.crossbow.result.TestResultHandler;
import uk.ac.imperial.lsds.crossbow.result.TrainingResultHandler;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class Dataflow {
	
	private final static Logger log = LogManager.getLogger (Dataflow.class);
	
	public static int autoincrement = 0;
	
	private SubGraph head;
	
	private ExecutionContext context;
	
	int nsubgraphs; /* Number of subgraphs */ 
	int noperators; /* Number of operators */
	
	private int id = -1; /* Dataflow id */
	
	private Phase phase = null;
	
	private ITaskDispatcher dispatcher;
	private IResultHandler handler;
	
	ResultCollector collector;
	
	public Dataflow (SubGraph graph) {
		head = graph;
		context = null;
		count (); /* Initialise number of sub-graphs and number of operators */
		id = autoincrement++;
		
		dispatcher = null;
		handler = null;
		collector = null;
	}
	
	private void count () {
		
		if (head == null)
			throw new NullPointerException("error: the most upstream dataflow sub-graph is null");
		
		nsubgraphs = 0;
		noperators = 0;
		
		SubGraph next = head;
		while (next != null) {
			nsubgraphs += 1;
			noperators += next.numberOfOperators();
			next = next.getNext();
		}
	}
	
	public Dataflow setPhase (Phase phase) {
		this.phase = phase;
		return this;
	}
	
	public SubGraph getSubGraph () {
		return head;
	}
	
	public ExecutionContext getExecutionContext () {
		return context;
	}
	
	public int numberOfSubGraphs () {
		return nsubgraphs;
	}
	
	public int numberOfOperators () {
		return noperators;
	}
	
	public int totalNumberOfOperators () {
		return Operator.cardinality();
	}
	
	public int getId () {
		return id;
	}
	
	public Phase getPhase () {
		return phase;
	}
	
	public boolean isTest () {
		return phase.equals(Phase.CHECK);
	}
	
	public ITaskDispatcher getTaskDispatcher () {
		return dispatcher;
	}
	
	public IResultHandler getResultHandler () {
		return handler;
	}
	
	/* Returns number of subgraphs in dataflow */
	public void init (ExecutionContext context) {
		
		log.debug (String.format("Init dataflow %d (%d sub-graph%s, %d operator%s)", 
				id, 
				numberOfSubGraphs(), (numberOfSubGraphs() > 1) ? "s" : "", 
				numberOfOperators(), (numberOfOperators() > 1) ? "s" : "")
		);
		
		if (head == null)
			throw new NullPointerException("error: the most upstream dataflow sub-graph is null");
		
		this.context = context;
		
		/* 
		 * Handle dependencies between the dispatcher the result 
		 * handler:
		 * 
		 * The dispatcher requires access to the result handler,
		 * and the result handler access to the input buffers of
		 * the dispatcher.
		 */
		switch (phase) {
		case TRAIN: handler = new TrainingResultHandler (this); break;
		case CHECK: handler = new     TestResultHandler (this); break;
		}
		
		switch (ModelConf.getInstance().getDataset(phase).getType()) {
		case  BASIC: dispatcher = new            TaskDispatcher (this); break;
		case  LIGHT: dispatcher = new LightWeightTaskDispatcher (this); break;
		case RECORD: dispatcher = new    ResNet50TaskDispatcher (this); break;
		}
		
		handler = handler.setup();
		
		if (SystemConf.getInstance().getGPU())
			TheGPU.getInstance().setResultHandler(phase.getId(), handler.getResultSlots(), handler.numberOfSlots());
		
		collector = new ResultCollector (phase, handler);
		
		Thread thread = new Thread(collector);
		thread.setName(String.format("Result collector (%s)", phase));
		thread.start();
		
		SubGraph next = head;
		while (next != null) {
			next.init(this);
			next = next.getNext();
		}
	}
	
	public ResultCollector getResultCollector () {
		if (collector == null)
			throw new NullPointerException ("error: result collector is null");
		return collector;
	}
	
	public void dump () {
		StringBuilder s = new StringBuilder (String.format("=== [Dataflow: %d sub-graphs, type: %s] ===\n", nsubgraphs, phase));
		SubGraph next = head;
		while (next != null) {
			s.append(next.dump());
			next = next.getNext();
		}
		s.append("=== [End of dataflow dump] ===");
		System.out.println(s.toString());
	}
	
	public void export (String filename) {
		StringBuilder s = new StringBuilder();
		SubGraph next = head;
		while (next != null) {
			s.append(next.export());
			next = next.getNext();
		}
		try {
			PrintWriter out = new PrintWriter(filename);
			out.print(s.toString());
			out.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public void exportDot (String filename) {
		StringBuilder s = new StringBuilder();
		SubGraph next = head;
		while (next != null) {
			s.append(next.exportDot());
			s.append("\n");
			next = next.getNext();
		}
		try {
			PrintWriter out = new PrintWriter(filename);
			out.print(s.toString());
			out.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public void dumpMemoryRequirements () {
		SubGraph next = head;
		while (next != null) {
			next.dumpMemoryRequirements ();
			next = next.getNext ();
		}
	}
	
	public static int cardinality () {
		return autoincrement;
	}
}
