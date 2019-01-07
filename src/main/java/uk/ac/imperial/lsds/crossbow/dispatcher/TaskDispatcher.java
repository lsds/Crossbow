package uk.ac.imperial.lsds.crossbow.dispatcher;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.BatchFactory;
import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.Dataset;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.dataset.DatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.result.IResultHandler;
import uk.ac.imperial.lsds.crossbow.task.Task;
import uk.ac.imperial.lsds.crossbow.task.TaskFactory;
import uk.ac.imperial.lsds.crossbow.task.TaskQueue;
import uk.ac.imperial.lsds.crossbow.types.DatasetFileType;
import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class TaskDispatcher implements ITaskDispatcher {
	
	private final static Logger log = LogManager.getLogger (TaskDispatcher.class);
	
	private Dataflow dataflow;
	
	private SubGraph graph;
	private TaskQueue queue;
	
	private IResultHandler handler;
	
	private Phase phase;
	private boolean test;
	
	/* Model configuration */
	
	private Dataset dataset;
	
	private VirtualCircularDataBuffer [] circularBuffer;
	
	private MappedDataBuffer [] buffer;
	
	private int [] tasksize;
	
	private long [] _tasksize = new long [2];
	
	private long [] accumulated;
	
	private long [] f; /* free  pointers */
	private long [] p; /* start pointers */
	private long [] q; /* end   pointers */
	
	private long [] _p, _q; /* Temporary copies for `p` and `q` used in method `newTaskFor()` */
	
	private int [] bid;
	
	private long [] idx;
	
	private boolean first = true;
	
	private int  w0, w1;
	private long _start;
	
	private boolean scheduleDirectly;
		
	public TaskDispatcher (Dataflow df) {
		
		dataflow = df;
		
		graph = dataflow.getSubGraph();
		queue = dataflow.getExecutionContext().getExecutorQueue();
		
		handler = dataflow.getResultHandler();
		
		phase = dataflow.getPhase();
		test = dataflow.isTest();
		
		if (ModelConf.getInstance ().getDataset (phase).getType () != DatasetType.BASIC)
			throw new IllegalStateException ("Invalid dataset in task dispatcher");
		
		dataset = (Dataset) ModelConf.getInstance ().getDataset (phase);
		
		circularBuffer = new VirtualCircularDataBuffer [2];
		
		tasksize = ModelConf.getInstance().getTaskSize ();
		
		circularBuffer[0] = new VirtualCircularDataBuffer (dataset.getExamples (), dataset.getExamplesCapacity (), (long) tasksize[0] * (long) SystemConf.getInstance().getTaskQueueSizeLimit());
		circularBuffer[1] = new VirtualCircularDataBuffer (dataset.getLabels   (), dataset.getLabelsCapacity   (), (long) tasksize[1] * (long) SystemConf.getInstance().getTaskQueueSizeLimit());		
		
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
		
		bid = new int [2];
		
		idx = new long [2];
		
		/* Window start and end pointer */
		w0 = w1 = 0;
		_start = 0;
		
		/* Does the dispatcher skip the task queue? */
		scheduleDirectly = SystemConf.getInstance().useDirectScheduling();
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
		
		/* 
		 * Set buffer id before we slide the window, since `q` can point to the end of a dataset file.
		 * However this causes problems when the circular buffer wraps.
		 */
		for (int i = 0; i < 2; ++i)
			bid [i] = w1;
		
		/* 
		 * Slide window before we normalise `p` and `q`. `w1` may change, sliding in a new file. 
		 * If this is the case, check if the circular buffer has wrapped.
		 */
		slideWindow ();
		
		/* log.info(String.format("Task %013d-%013d", _p[0], _q[0])); */
		
		for (int i = 0; i < 2; ++i) {
			
			/* `p` and `q` are normalised */
			if (bid [i] != circularBuffer [i].getAddressTranslator ().translate (i, _p, _q)) {
				
				System.err.println(String.format("error: invalid address translation: (i = %d) %d-%d does not correspond to file id %d; extected value is %d", 
						i, _p[i], _q[i], bid[i], circularBuffer [i].getAddressTranslator ().translate (i, _p, _q)));
				System.exit(1);
			}
		}
		
		buffer [0] = dataset.getExamples (bid [0]);
		buffer [1] = dataset.getLabels   (bid [1]);
		
		if (scheduleDirectly) {
			
			TheGPU.getInstance().schedule(graph.getId(), taskid, buffer[0], _p[0], _q[0], buffer[1], _p[1], _q[1], f, (test ? 1 : 0), bound);
		} 
		else {
			
			batch = BatchFactory.newInstance (taskid, bound, buffer, tasksize, _p, _q, f, dataflow.totalNumberOfOperators());
			
			task = TaskFactory.newInstance (taskid, graph, batch, handler, null);
			task.setValidationTask(test);
			
			queue.add(task);
		}
	}
	
	private void slideWindow () {
		
		/* If the dataset consists of a single file, do nothing */
		if (dataset.numberOfPartitions() == 1)
			return;
		
		long start = circularBuffer[0].getNormalisedStartPointer ();
		if (start >= _start) {
			while (start > circularBuffer[0].getAddressTranslator().getEndPointer (w0)) {
				/* Slide-out old dataset file */
				/* log.info(String.format("Slide-out file id %2d", w0)); */
				DatasetMemoryManager.getInstance().slideOut (phase.getId(), w0);
				/* Increment file pointer */
				w0 = (w0 + 1) % dataset.numberOfPartitions();
			}
		} else {
			
			while (_start <= circularBuffer[0].capacity() && w0 < dataset.numberOfPartitions()) {
				
				/* log.info(String.format("Previous start %d start %d w0 %d", _start, start, w0)); */
				
				/* Slide-out old dataset file */
				/* log.info(String.format("Slide-out file id %2d", w0)); */
				DatasetMemoryManager.getInstance().slideOut (phase.getId(), w0);
				
				/* Increment file pointer */
				w0 = (w0 + 1);
				
				/* Increment start pointer */
				if (w0 < dataset.numberOfPartitions())
					_start = circularBuffer[0].getAddressTranslator().getEndPointer (w0);
			}
			
			/* log.info("Again..."); */
			
			w0 %= dataset.numberOfPartitions();
			
			/* Continue as normal */
			while (start > circularBuffer[0].getAddressTranslator().getEndPointer (w0)) {
				
				/* log.info(String.format("Previous start %d start %d w0 %d", _start, start, w0)); */
				
				/* Slide-out old dataset file */
				/* log.info(String.format("Slide-out file id %2d", w0)); */
				DatasetMemoryManager.getInstance().slideOut (phase.getId(), w0);
				/* Increment file pointer */
				w0 = (w0 + 1) % dataset.numberOfPartitions();
			}
		}
		
		if (_q[0] == circularBuffer[0].getAddressTranslator ().getEndPointer (w1)) {
			
			w1 = (w1 + 1) % dataset.numberOfPartitions();
			
			/* Slide-in dataset file */
			log.info(String.format("Slide-in file id %2d", w1));
			DatasetMemoryManager.getInstance().slideIn (phase.getId(), w1);
			
			/* Update (11/12/2017)
			 * 
			 * Assign new addresses to buffers (since they may be mapped again in memory...) 
			 */
			dataset.getExamples(w1).setAddress (DatasetMemoryManager.getInstance().address (phase.getId(), DatasetFileType.EXAMPLES.getId(), w1));
			dataset.getLabels  (w1).setAddress (DatasetMemoryManager.getInstance().address (phase.getId(), DatasetFileType.LABELS.getId(),   w1));
		}
		
		/* Track previous start pointer */
		_start = start;
		
		/* log.info(String.format("%013d-%013d: %04d-%04d", start, _q[0], w0, w1)); */
	}
	
	public VirtualCircularDataBuffer [] getBuffers () {
		return circularBuffer;
	}
	
	public void dispatch (Batch batch, Integer replicaId) {
		throw new UnsupportedOperationException ("error: unsupported task dispatcher method call");
	}
}
