package uk.ac.imperial.lsds.crossbow.dispatcher;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.BatchFactory;
import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.LightWeightDataset;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;
import uk.ac.imperial.lsds.crossbow.result.IResultHandler;
import uk.ac.imperial.lsds.crossbow.task.Task;
import uk.ac.imperial.lsds.crossbow.task.TaskFactory;
import uk.ac.imperial.lsds.crossbow.task.TaskQueue;
import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

/*
 * Light-weight task and data management
 */
public class LightWeightTaskDispatcher implements ITaskDispatcher {
	
	private final static Logger log = LogManager.getLogger (LightWeightTaskDispatcher.class);
	
	private Dataflow dataflow;
	
	private SubGraph graph;
	private TaskQueue queue;
	
	private IResultHandler handler;
	
	private Phase phase;
	private boolean test;
	
	/* Model configuration */
	
	private LightWeightDataset dataset;
	
	private VirtualCircularDataBuffer [] circularBuffer;
	
	private MappedDataBuffer [] buffer;
	
	private int [] tasksize;
	
	private long [] _tasksize = new long [2];
	
	private long [] accumulated;
	
	private long [] f; /* free  pointers */
	private long [] p; /* start pointers */
	private long [] q; /* end   pointers */
	
	private long [] _p, _q; /* Temporary copies for `p` and `q` used in method `newTaskFor()` */
	
	private long [] idx;
	
	private boolean first = true;
	
	public LightWeightTaskDispatcher (Dataflow df) {
		
		dataflow = df;
		
		graph = dataflow.getSubGraph();
		queue = dataflow.getExecutionContext().getExecutorQueue();
		
		handler = dataflow.getResultHandler();
		
		phase = dataflow.getPhase();
		test = dataflow.isTest();
		
		if (ModelConf.getInstance ().getDataset (phase).getType () != DatasetType.LIGHT)
			throw new IllegalStateException ("Invalid dataset in task dispatcher");
		
		dataset = (LightWeightDataset) ModelConf.getInstance ().getDataset (phase);
		
		circularBuffer = new VirtualCircularDataBuffer [2];
		
		tasksize = ModelConf.getInstance().getTaskSize ();
		
		circularBuffer[0] = new VirtualCircularDataBuffer (null, dataset.getExamplesCapacity ());
		circularBuffer[1] = new VirtualCircularDataBuffer (null, dataset.getLabelsCapacity   ());		
		
		/* Pointers to actual mapped data buffers for examples (0) and labels (1) */
		buffer = new MappedDataBuffer [2];
		
		accumulated = new long [2];
		
		f = new long [2];
		p = new long [2];
		q = new long [2];
		
		for (int i = 0; i < 2; i++) {
			
			accumulated[i] = 0;
			
			f[i] = 0;
			p[i] = 0;
			q[i] = tasksize [i];
		}
		
		_p = new long [2];
		_q = new long [2];
		
		idx = new long [2];
		
		buffer [0] = dataset.getExamples ();
		buffer [1] = dataset.getLabels   ();
	}
	
	public void dispatchNext (int taskid, int bound) {
		
		for (int i = 0; i < 2; ++i)
			while ((idx[i] = circularBuffer[i].shift(tasksize[i])) < 0) {
				Thread.yield();
			}
		
		assemble (taskid, bound);
	}
	
	private void assemble (int taskid, int bound) {
		
		if (first) {
			for (int i = 0; i < 2; i++) {
				p[i] = 0;
				q[i] = tasksize[i];
			}
			first = false;
		}
		
		for (int i = 0; i < 2; i++) {
			f[i] = circularBuffer[i].normalise (q[i]);
			f[i] = (f[i] == 0) ? circularBuffer[i].capacity () : f[i];
			f[i]--;
		}
		
		/* Launch task */
		newTaskFor (taskid, bound);
			
		for (int i = 0; i < 2; i++) {
			p[i] += tasksize[i];
			q[i] += tasksize[i];
		}
	}
		
	private void newTaskFor (int taskid, int bound) {
		
		Task task;
		Batch batch;
		
		for (int i = 0; i < 2; i++) {
			_p[i] = p[i] % circularBuffer[i].capacity();
			_q[i] = q[i] % circularBuffer[i].capacity();
		}
		
		if (log.isDebugEnabled()) {
			
			for (int i = 0; i < 2; i++)
				_tasksize[i] = (_q[i] <= _p[i]) ? (_q[i] + circularBuffer[i].capacity()) - _p[i] : _q[i] - _p[i];
				
			log.debug(
				String.format("[DBG] Phase %8s subgraph %d task %6d [%10d, %10d) ([%10d, %10d)), free %10d (%10d) size %10d (%10d)", 
					phase.toString(), graph.getId(), taskid, _p[0], _q[0], _p[1], _q[1], f[0], f[1], _tasksize[0], _tasksize[1]));
		}
		
		for (int i = 0; i < 2; i++)
			if (_q[i] <= _p[i])
				_q[i] += circularBuffer[i].capacity();
		
		/* Check free pointers */
		if (f[0] < 0 || f[1] < 0) {
			
			System.err.println(String.format("error: negative free pointer for task %d ([0]=%d, [1]=%d)", taskid, f[0], f[1]));
			System.exit(1);
		}
		
		dataset.translate (_p, _q, f);
		
		batch = BatchFactory.newInstance (taskid, bound, buffer, tasksize, _p, _q, f, dataflow.totalNumberOfOperators());
		
		task = TaskFactory.newInstance (taskid, graph, batch, handler, null);
		
		task.setValidationTask(test);
		
		queue.add(task);
	}
	
	public void dispatch (Batch batch, Integer replicaId) {
		throw new UnsupportedOperationException ("error: unsupported task dispatcher method call");
	}

	public VirtualCircularDataBuffer [] getBuffers () {
		return circularBuffer;
	}
}