package uk.ac.imperial.lsds.crossbow;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.dispatcher.ITaskDispatcher;
import uk.ac.imperial.lsds.crossbow.dispatcher.TransientTaskDispatcher;
import uk.ac.imperial.lsds.crossbow.kernel.KernelMemoryRequirements;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.task.Task;
import uk.ac.imperial.lsds.crossbow.types.DependencyType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.utils.CrossbowArrayList;
import uk.ac.imperial.lsds.crossbow.utils.CrossbowLinkedList;

public class SubGraph {
	
	private final static Logger log = LogManager.getLogger (SubGraph.class);
	
	public static int autoincrement = 0;
	
	/* There is only one most upstream (`head`) and most downstream (`tail`) per subgraph */
	private DataflowNode head, tail;
	
	/* Subgraphs are wired in a chain: there is only one `prev` and `next` subgraph */
	private SubGraph prev, next;
	
	private ITaskDispatcher dispatcher;
	
	private Dataflow dataflow;
	
	/*
	 * The  inputShape is the shape of the `head`;
	 * the outputShape is the shape of the `tail`.
	 */
	private Shape inputShape, outputShape;
	
	private ModelAccess access;
	
	private int order, branch;
	
	private int id = -1;
	
	public SubGraph (DataflowNode node) {
		
		head = tail = node;
		
		prev = next = null;
		
		dispatcher = null;
		dataflow = null;
		
		inputShape = outputShape = null;
		
		access = ModelAccess.NA;
		
		order = 0;
		branch = 0;
		
		topologicalSort (); /* Initialise number of operators */
		
		/*
		t_level ();
		b_level ();
		branchOut ();
		*/
		
		id = autoincrement++;
	}
	
	private void topologicalSort () {
		
		if (head == null)
			throw new NullPointerException("error: the most upstream dataflow node in sub-graph is null");
		
		/* Depth-first topological sort */
		CrossbowLinkedList<DataflowNode> list = new CrossbowLinkedList<DataflowNode> ();
		head.visit(list);
		
		/* Configure execution order */
		order = 0;
		Iterator<DataflowNode> iterator = list.iterator();
		DataflowNode curr = head;
		while (iterator.hasNext()) {
			
			DataflowNode node = iterator.next ();
			curr = curr.setNextInTopology (node);
			
			node.setOrder (order ++);
			node.finalise ();
		}
	}
	
	private void t_level () {
		int max;
		DataflowNode node = head;
		while (node != null) {
			max = 0;
			if (node.getPreviousList () != null) {
				for (DataflowNode upstream: node.getPreviousList()) {
					if (upstream.getLevel () < 0)
						throw new IllegalStateException ();
					if (max < upstream.getLevel () + 1)
						max = upstream.getLevel () + 1;
				}
			}
			node.setLevel (max);
			node = node.getNextInTopology();
		}
	}
	
	/*
	private void b_level () {
		int max;
		DataflowNode node;
		DataflowNode tail;
		
		node = head;
		tail = null;
		while (node != null) {
			tail = node;
			node = node.getNextInTopology();
		}
		
		node = tail;
		while (node != null) {
			max = 0;
			if (node.getNextList () != null) {
				for (DataflowNode downstream: node.getNextList()) {
					if (downstream.getLevel () < 0)
						throw new IllegalStateException ();
					if (max < downstream.getLevel ())
						max = downstream.getLevel ();
				}
			}
			node.setLevel (max + 1);
			node = node.getPreviousInTopology();
		}
	}
	*/
	
	private void branchOut () {
		
		DataflowNode node = head;
		while (node != null) {
			
			if (node.getPreviousList () == null) { 
				/* Assign label to most upstream node */
				head.setLabel (branch++);
			}
			
			if (node.getNextList () != null) {
				/* Assign labels to downstream nodes, incrementing by 1 on every branch */
				/*
				DataflowNode theOne = null;
				for (DataflowNode downstream: node.getNextList()) {
					if ((theOne == null) || (theOne.getLevel() > downstream.getLevel()))
							theOne = downstream;
				}
				*/
				boolean first = true;
				for (DataflowNode downstream: node.getNextList()) {
					/*
					if (downstream == theOne) {
						downstream.setLabel(node.getLabel());
					}
					*/
					if (first) {
						downstream.setLabel(node.getLabel());
						first = false;
					}
					else {
						/*
						 * Note that some branches just denote
						 * dependencies between kernels.
						 * 
						 * For example:
						 * 
						 * A -> B
						 * A -> C
						 * B -> C
						 * 
						 * In such cases, we don't increment `maxlabels`
						 */
						if (downstream.getPreviousList().size() == 1)
							downstream.setLabel(branch ++);
					}
				}
			}
			node = node.getNextInTopology();
		}
	}
	
	public DataflowNode getDataflowNode () {
		return head;
	}
	
	public int numberOfOperators () {
		return order;
	}
	
	public int getBranchFactor () {
		if (branch == 0)
			return 1;
		return branch;
	}
	
	public String getName () {
		return String.format("sub-graph %d", id);
	}
	
	public SubGraph getNext () {
		return next;
	}
	
	public SubGraph getPrevious () {
		return prev;
	}
	
	public void setPrevious (SubGraph graph) {
		prev = graph;
	}
	
	public SubGraph connectTo (SubGraph graph) {
		next = graph;
		graph.setPrevious(this);
		return graph;
	}
	
	public ModelAccess getModelAccessType () {
		return access;
	}
	
	public int getId () {
		return id;
	}
	
	public void process (Batch batch, Integer replicaId, Task task, boolean GPU) {
		
		if (! GPU) {
			
			Model model = null;
			if (replicaId != null)
				model = dataflow.getExecutionContext().getModelManager().getModel(replicaId);
			
			DataflowNode next = head;
			while (next != null) {
				
				/* Get the current operator */
				Operator p = next.getOperator();
				
				/* Get the operators of the upstream nodes with next.getPreviousOperators() */
				p.getKernel().compute (next.getPreviousOperators (), batch, model, task);
				
				if (log.isDebugEnabled())
					p.computeChecksum (batch.getOutput(p.getId()));
				
				next = next.getNextInTopology();
			}
		} 
		else {
			/* Schedule task on the GPU */
			TheGPU.getInstance().execute(getId(), batch, replicaId, task);
		}
	}
	
	public ITaskDispatcher getTaskDispatcher () {
		return dispatcher;
	}
	
	public Dataflow getDataflow () {
		return dataflow;
	}
	
	public boolean isMostUpstream () {
		return (getPrevious() == null);
	}
	
	public Shape getInputShape () {
		return inputShape;
	}
	
	public Shape getOutputShape () {
		return outputShape;
	}
	
	public void init (Dataflow dataflow) {
		
		if (head == null)
			throw new NullPointerException("error: the most upstream dataflow node in sub-graph is null");
		
		log.debug(String.format("Init %s (%d operator%s)", 
				getName(), numberOfOperators(), (numberOfOperators() > 1) ? "s" : ""));
		
		this.dataflow = dataflow;
		
		/* Initialise task dispatcher */
		if (! isMostUpstream ())
			dispatcher = new TransientTaskDispatcher (this);
		
		Phase phase = dataflow.getPhase();
		
		if (isMostUpstream ()) {
			inputShape = ModelConf.getInstance().getInputShape (phase);
		} 
		else {
			inputShape = getPrevious().getOutputShape();
		}
		
		log.debug("Input shape is " + inputShape);
		
		Model model = dataflow.getExecutionContext().getModel();
		
		DataflowNode next = head;
		
		while (next != null) {
			
			Operator operator = next.getOperator();
			/* Set dataflow node */
			operator.setDataflowNode (phase, next);
			
			/* Get the shape of the operators of the upstream nodes connected to `next` */
			Operator [] upstream = next.getPreviousOperators();
			
			Shape [] shape;
			
			if (upstream == null) {
				shape = new Shape [1];
				shape[0] = inputShape;
			}
			else {
				shape = new Shape [upstream.length];
				
				for (int i = 0; i < upstream.length; ++i) {
					shape[i] = upstream[i].getOutputShape();
				}
			}
			
			operator.init (shape, model);
			
			ModelAccess other = operator.getModelAccessType();
			if (access.compareTo(other) < 0)
				access = other;
			
			next = next.getNextInTopology();
		}
		
		/* The output shape of the sub-graph is the output shape of its tail operator. */
		outputShape = tail.getOperator().getOutputShape();
		
		/* Try to optimise memory plan */
		tryOptimise ();
	}
	
	private void tryOptimise () {
		
		if (! SystemConf.getInstance().tryReuseMemory())
			return;
			
		DataflowNode [] plan = MemoryPlannerV2.analyse (this);

		if (log.isInfoEnabled()) {
			/* Dump analysis on screen */
			StringBuilder s = new StringBuilder ();
			s.append(String.format("=== [%s memory re-use plan (%d operators)] ===\n", getName(), numberOfOperators()));
			DataflowNode next = head;
			while (next != null) {
				int idx = next.getOrder();
				s.append(String.format("%4d: %3s (%s)\n", idx, (plan[idx] == null) ? "new" : String.format("%3d", plan[idx].getOrder()), next.getOperator().getName()));
				next = next.getNextInTopology();
			}
			s.append(String.format("=== [End of %s's memory re-use plan] ===", getName()));
			System.out.println(s);
		}
		
		/* Iterate over nodes and set inPlaceComputations */
		DataflowNode next = head;

		while (next != null) {

			DataflowNode node = plan [next.getOrder()];
			if (node != null)
				next.getOutputBufferFrom (node);

			next = next.getNextInTopology();
		}
		
		return;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register dataflow sub-graph #%d", id));
		
		__register_dataflow_graph (); __register_dataflow_loss_operator (); __register_dataflow_accuracy_operator (); __register_dataflow_datatransform_operator ();
		__register_dataflow_peers ();
		
		/* Register memory plan */
		if (SystemConf.getInstance().tryReuseMemory())
			__register_dataflow_memoryplan ();
		
		/* Register operator dependencies */
		/* __register_dataflow_dependency_graph (); */
	}
	
	private void __register_dataflow_graph () {
		
		int [] ops = new int [numberOfOperators()];
		
		int i = 0;
		DataflowNode next;
		
		/* Iterate over dataflow in topological order */
		next = head;
		while (next != null) {
			
			log.debug(String.format("%3d: %3d or %s", i, next.getOperator().getId(), next.getOperator().getName()));
			ops[i++] = next.getOperator().getId();
			next = next.getNextInTopology();
		}
		
		TheGPU.getInstance().setDataflowGraph (id, ops);
		
		/* Iterate once again, wiring operators together */
		next = head;
		while (next != null) {
			
			CrossbowArrayList<DataflowNode> prevList = next.getPreviousList ();
			
			if (prevList != null) {
				
				int [] upstream = new int [prevList.size()];
				for (i = 0; i < upstream.length; ++i)
					upstream [i] = prevList.get(i).getOrder();
				
				TheGPU.getInstance().setDataflowUpstreamNeighbours (id, next.getOrder(), upstream);
			} 
			
			CrossbowArrayList<DataflowNode> nextList = next.getNextList ();
			
			if (nextList != null) {
				
				int [] downstream = new int [nextList.size()];
				for (i = 0; i < downstream.length; ++i)
					downstream [i] = nextList.get(i).getOrder();
				
				TheGPU.getInstance().setDataflowDownstreamNeighbours (id, next.getOrder(), downstream);
			}
			
			next = next.getNextInTopology();
		}
	}
	
	private void __register_dataflow_loss_operator () {
		
		log.debug(String.format("Register dataflow sub-graph #%d's loss operator", id));
		
		Operator op = null;
		DataflowNode next = head;
		
		while (next != null) {
			if (next.getOperator().getKernel().isLossKernel()) {
				if (op != null)
					throw new IllegalStateException("error: a dataflow should contain at most one loss operator");
				op = next.getOperator();
			}
			next = next.getNextInTopology();
		}
		
		if (op != null)
			TheGPU.getInstance().setDataflowLossOperator(id, op.getId());
	}
	
	private void __register_dataflow_accuracy_operator () {
		
		log.debug(String.format("Register dataflow sub-graph #%d's accuracy operator", id));
		
		Operator op = null;
		DataflowNode next = head;
		
		while (next != null) {
			if (next.getOperator().getKernel().isAccuracyKernel()) {
				if (op != null)
					throw new IllegalStateException("error: a dataflow should contain at most one accuracy operator");
				op = next.getOperator();
			}
			next = next.getNextInTopology();
		}
		
		if (op != null)
			TheGPU.getInstance().setDataflowAccuracyOperator(id, op.getId());
	}
	
	private void __register_dataflow_datatransform_operator () {
		
		log.debug(String.format("Register dataflow sub-graph #%d's data transform operator", id));
		
		Operator op = null;
		DataflowNode next = head;
		
		while (next != null) {
			if (next.getOperator().getKernel().isDataTransformationKernel()) {
				if (op != null)
					throw new IllegalStateException("error: a dataflow should contain at most one data transform operator");
				op = next.getOperator();
			}
			next = next.getNextInTopology();
		}
		
		if (op != null)
			TheGPU.getInstance().setDataflowDataTransformOperator(id, op.getId());
	}
	
	private void __register_dataflow_peers () {
		
		log.debug(String.format("Register dataflow sub-graph #%d's peerings", id));
		
		int [] peers = new int [numberOfOperators()];
		Arrays.fill(peers, -1);
		
		int i = 0;
		DataflowNode next = head;
		
		while (next != null) {
			if (next.getOperator().isGradient())
				peers[i] = next.getOperator().getPeer().getId();
			i++;
			next = next.getNextInTopology();
		}
		
		TheGPU.getInstance().setDataflowPeers (id, peers);
	}
	
	private void __register_dataflow_memoryplan () {
		
		DataflowNode next = head;
		
		while (next != null) {
			
			int provider = next.getOutputBufferFromElsewhere () ? next.getOutputBufferDonor().getOrder() : -1;
			int position = next.getOutputBufferFromPosition ();
			TheGPU.getInstance().setDataflowMemoryPlan (id, next.getOrder(), provider, position);
			
			next = next.getNextInTopology();
		}
	}
	
	private void __register_dataflow_dependency_graph () {
		
		DataflowNode next = head;
		
		while (next != null) {
			
			TheGPU.getInstance().setDataflowStream (id, next.getOrder(), next.getLabel());
			CrossbowArrayList<DataflowNode> upstreams = next.getPreviousList();
			if (upstreams != null) {
				/* Iterate over upstream dataflow nodes */
				for (DataflowNode prev: upstreams) {
					if (next.getLabel() != prev.getLabel()) {
						/* Operator `prev` must end before operator `next` starts */
						TheGPU.getInstance().setDataflowDependency(id, next.getOrder(), DependencyType.END_BEFORE_START.getId(), prev.getOrder(), true);
					}
				}
			}
			
			next = next.getNextInTopology();
		}
	}
	
	public String dump () {
		StringBuilder s = new StringBuilder (String.format("[%s: %d operators (access: %s)] ", 
				getName(), numberOfOperators(), getModelAccessType()));
		DataflowNode next = head;
		while (next != null) {
			s.append(String.format("%s -> ", next.getOperator().getName()));
			next = next.getNextInTopology();
		}
		s.append("null\n");
		return s.toString();
	}
	
	public String export () {
		
		StringBuilder s = new StringBuilder();
		
		int opid, executionOrder;
		String name, type, children, parents;
		int [] ids;
		Operator [] upstreams, downstreams;
		
		DataflowNode curr = head;
		while (curr != null) {
			
			opid = curr.getOperator().getId();
			name = curr.getOperator().getName();
			type = curr.getOperator().getKernel().getClass().getSimpleName();
			
            executionOrder = curr.getOrder();
			
			upstreams = curr.getPreviousOperators();
			parents = "[]";
			if (upstreams != null) {
				ids = new int [upstreams.length];
				for (int i = 0 ; i < upstreams.length; i++) {
					ids[i] = upstreams[i].getId();
				}
				parents = Arrays.toString(ids);
			}
			
			downstreams = curr.getNextOperators();
			children = "[]";
			if (downstreams != null){
				ids = new int [downstreams.length];
				for (int i = 0 ; i < downstreams.length; i++) {
					ids[i] = downstreams[i].getId();
				}
				children = Arrays.toString(ids);
			}
			
			s.append(String.format("%d\t%d\t%s\t%s\t%s\t%s\n", opid, executionOrder, name, type, children, parents));
			/* Move to next node */
			curr = curr.getNextInTopology();
		}
		return s.toString();
	}
	
	public String exportDot () {
		DataflowNode node;
		StringBuilder s = new StringBuilder();
		s.append(String.format("digraph subgraph%d {\n\n", getId()));
		/* Create nodes */
		node = head;
		while (node != null) {
			s.append(node.exportDot());
			node = node.getNextInTopology();
		}
		s.append("\n");
		/* Create edges */
		node = head;
		while (node != null) {
			if (node.getNextList() != null) {
				for (DataflowNode downstream: node.getNextList()) {
					s.append(String.format("\tn%d -> n%d\n", node.getOrder(), downstream.getOrder()));
				}
			}
			node = node.getNextInTopology();
		}
		s.append("}\n");
		return s.toString();
	}
	
	public void dumpMemoryRequirements () {
		
		StringBuilder s = new StringBuilder (String.format("=== [%s memory requirements (%d operators)] ===\n", 
				getName(), numberOfOperators()));
		
		KernelMemoryRequirements total = new KernelMemoryRequirements ();
		
		s.append(String.format("%5s:\t%9s\t%9s\t%9s\t%9s (%s)\n", "Id", "Output", "Model", "CPU vars", "GPU vars", "Name"));
		DataflowNode next = head;
		while (next != null) {
			
			KernelMemoryRequirements requirements = next.getOperator().getKernel().getKernelMemoryRequirements();
			
			s.append(String.format("%5d:\t%s\t%s\t%s\t%s (%s)\n", 
					next.getOrder(),
					KernelMemoryRequirements.bytesToString (requirements.getOutputMemoryRequirements   (next.getOutputBufferFromElsewhere())),
					KernelMemoryRequirements.bytesToString (requirements.getModelMemoryRequirements    ()),
					KernelMemoryRequirements.bytesToString (requirements.getLocalCPUMemoryRequirements ()),
					KernelMemoryRequirements.bytesToString (requirements.getLocalGPUMemoryRequirements ()),
					next.getOperator().getName()
					));
			
			total.incOutputMemoryRequirements   (requirements.getOutputMemoryRequirements   (next.getOutputBufferFromElsewhere()));
			total.incModelMemoryRequirements    (requirements.getModelMemoryRequirements    ());
			total.incLocalCPUMemoryRequirements (requirements.getLocalCPUMemoryRequirements ());
			total.incLocalGPUMemoryRequirements (requirements.getLocalGPUMemoryRequirements ());
			
			next = next.getNextInTopology();
		}
		
		s.append(String.format("total:\t%s\t%s\t%s\t%s\n", 
				KernelMemoryRequirements.bytesToString (total.getOutputMemoryRequirements   (false)),
				KernelMemoryRequirements.bytesToString (total.getModelMemoryRequirements    ()),
				KernelMemoryRequirements.bytesToString (total.getLocalCPUMemoryRequirements ()),
				KernelMemoryRequirements.bytesToString (total.getLocalGPUMemoryRequirements ())
				));
		
		s.append(String.format("=== [End of %s's memory requirements dump] ===", 
				getName()));
		System.out.println(s.toString());
	}
	
	public static int cardinality () {
		return autoincrement;
	}
}
