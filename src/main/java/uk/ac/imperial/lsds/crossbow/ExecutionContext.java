package uk.ac.imperial.lsds.crossbow;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.device.TheCPU;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.device.dataset.DatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.device.dataset.LightWeightDatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.device.random.RandomGenerator;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.ModelManager;
import uk.ac.imperial.lsds.crossbow.processor.TaskProcessorPool;
import uk.ac.imperial.lsds.crossbow.task.TaskQueue;
import uk.ac.imperial.lsds.crossbow.types.DurationUnit;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class ExecutionContext {
	
	private final static Logger log = LogManager.getLogger (ExecutionContext.class);

	private Dataflow [] dataflows;
	
	private Model theModel;
	private ModelManager modelManager;
	
	private final int M = 2; /* Number of devices */
	private int N; /* Number of unique operators */
	private int [][] matrix;
	
	private TaskQueue queue;
	private TaskProcessorPool workerPool;
	private Executor executor;
	
	private WorkClock workClock;
	
	private PerformanceMonitor monitor;
	
	public ExecutionContext (Dataflow [] dataflows) {
		this.dataflows = dataflows;
	}
	
	public Dataflow [] getDataflows () {
		return dataflows;
	}
	
	public Dataflow getDataflow (Phase type) {
		switch (type) {
		case TRAIN: return dataflows[0];
		case CHECK: return dataflows[1];
		default:
			throw new IllegalArgumentException ("error: invalid phase type");
		}
	}
	
	public Model getModel () {
		return theModel;
	}
	
	public ModelManager getModelManager () {
		return modelManager;
	}
	
	public int [][] getThroughputMatrix () {
		return matrix;
	}
	
	public TaskQueue getExecutorQueue() {
		return queue;
	}
	
	public int getExecutorQueueSize() {
		return queue.size();
	}
	
	public TaskProcessorPool getTaskProcessorPool () {
		return workerPool; 
	}
	
	public WorkClock getWorkClock () {
		return workClock;
	}
	
	public void init () {
		
		TheCPU.getInstance().init ();
		
		/* Bind the main thread (this) to CPU core 0 */
		TheCPU.getInstance().bind(0);
		
		/* Initialise the (Open)BLAS library only if CPU mode is enabled */
		if (SystemConf.getInstance().getCPU())
			BLAS.getInstance().init();
		
		TheGPU.getInstance().init();
		
		/* Load random generator library */
		RandomGenerator.getInstance().load();
		RandomGenerator.getInstance().init(SystemConf.getInstance().getRandomSeed());
		
		queue = new TaskQueue ();
		
		theModel = new Model ();
		
		/* Initialise data sets (map and register) */
		ModelConf.getInstance().init();
		
		for (int i = 0; i < dataflows.length; ++i)
			if (dataflows[i] != null)
				dataflows[i].init(this);
		
		theModel.finalise(Operator.cardinality());
		
		modelManager = new ModelManager (theModel);
		
		TheGPU.getInstance().register (this);
		
		workClock = new WorkClock (0, ModelConf.getInstance().getWpc(), Integer.MAX_VALUE);
		
		N = SubGraph.cardinality();
		
		matrix = new int [M][N];
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				matrix [i][j] = 1;
			}
		}
		
		workerPool = new TaskProcessorPool (this);
		
		executor = Executors.newCachedThreadPool();
		queue = workerPool.start(executor);
		
		monitor = new PerformanceMonitor(this);
		
		Thread thread = new Thread(monitor);
		thread.setName("Monitor");
		thread.start();
		
		/* Pass monitor to model manager in case of auto-tuning */
		if (SystemConf.getInstance().autotuneModels())
			modelManager.setPerformanceMonitor(monitor);
		
		return;
	}
	
	public void trainAndTest (int count, TrainingUnit unit) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid;
		int lowerBound;
		
		int N = (unit == TrainingUnit.EPOCHS) ? (ModelConf.getInstance().numberOfTasksPerEpoch() * count) : count;
		int M = ModelConf.getInstance().numberOfTestTasks();
		
		int step = ModelConf.getInstance().getTestInterval();
		
		if (step % ModelConf.getInstance().getWpc() != 0) {
			while (step % ModelConf.getInstance().getWpc() != 0) {
				step++;
			}
		}
		
		if (step <= 0)
			throw new IllegalArgumentException("error: test interval must greater than 0");
		
		if (N < step) {
			log.warn(String.format("Test interval greater than number of training tasks"));
			/* Reset to `tasks`. Testing will occur at end of training */
			step = N;
		}
		
		boolean inline;
		if (N == step) {
			log.info(String.format("Train for %d tasks", N));
			inline = false;
		} else { /* N > step */
			log.info(String.format("Train for %d tasks; test every %d tasks", N, step));
			inline = true;
		}
		
		CountDownLatch l0 = new CountDownLatch (N);
		CountDownLatch l1 = null;
		
		dataflows[0].getResultCollector().setCountDownLatch(l0);
		
		log.info(String.format("Start scheduling tasks at %d", System.nanoTime()));
		
		for (int i = 0; i < N; ++i) {
			
			taskid = workClock.incrementAndGetNext(bound);
			
			/*
			int [] steps = ModelConf.getInstance().getSolverConf().getStepValues();
            boolean found = false;
			for (int j = 0; j < steps.length; ++j) {
				if (taskid == steps[j]) {
					found = true;
					break;
				}
			}
			if (found) {
				log.info("Changing clock...");
				workClock.wpc = workClock.wpc / 2;
				// clock.incrementAndGetNext(null);
			}
			*/
			
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			
			if (inline && (((i + 1) % step) == 0)) {
				/* Launch test tasks */
				if (l1 != null) {
					log.debug("Waiting for testing tasks to finish...");
					l1.await();
				}
				log.info(String.format("Start test phase at %d", System.nanoTime()));
				l1 = new CountDownLatch (M);
				/* Set new count-down latch to test result collector */
				
				dataflows[1].getResultCollector().setCountDownLatch(l1);
				/* Launch test tasks */
				int clock = workClock.getClock();
				/* log.info(M + " test tasks with clock " + clock); */
				for (int id = 1; id <= M; ++id) {
					dataflows[1].getTaskDispatcher().dispatchNext(id, clock);
				}
			}
			
			/* Auto-tune number of models */
		}
		/* Await completion of training & test tasks */
		l0.await();
		if (inline) /* At least one set of test tasks is scheduled */
			l1.await();
		modelManager.synchroniseAll();
		if (N % ModelConf.getInstance().getWpc() == 0) {
			log.info("One last test at %d...", System.nanoTime());
			test();
		}
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void train (int count, TrainingUnit unit) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid;
		int lowerBound;
		
		int tasks = (unit == TrainingUnit.EPOCHS) ? (ModelConf.getInstance().numberOfTasksPerEpoch() * count) : count;
		
		log.info(String.format("Train for %d tasks", tasks));
		
		CountDownLatch latch = new CountDownLatch (tasks);
		dataflows[0].getResultCollector().setCountDownLatch(latch);
		
		for (int i = 0; i < tasks; ++i) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
		}
		/* Await task completion */
		latch.await();
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void trainForDuration (double duration, DurationUnit unit) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid = 0;
		int lowerBound;
		
		long seconds = 0;
		
		if (unit == DurationUnit.SECONDS) seconds = Math.round(duration * 1.0);
		else 
		if (unit == DurationUnit.MINUTES) seconds = Math.round(duration * 60.0);
		else 
		if (unit == DurationUnit.HOURS  ) seconds = Math.round(duration * 3600.0);
		
		log.info(String.format("Train for %d seconds", seconds));
		
		AtomicBoolean finish = new AtomicBoolean (false);
		
		long numberoftasks = 0L;
		
		CountDownTimer timer = new CountDownTimer (finish);
		timer.set(seconds);
		Thread thread = new Thread (timer);
		thread.setName("CountDownTimer");
		thread.start();
		
		while (! finish.get()) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			numberoftasks ++;
		}
		log.info(String.format("Received wake-up call after %d tasks", numberoftasks));
		/* Await completion of the last task dispatched. */
		while (dataflows[0].getResultCollector().getNumberOfTasks() < numberoftasks)
			LockSupport.parkNanos(1L);
		
		log.info(String.format("Trained for %d tasks", numberoftasks));
		
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void trainForDuration 
		(double duration, DurationUnit durationUnit, int count, TrainingUnit trainingUnit) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid = 0;
		int lowerBound;
		
		long seconds = 0;
		
		if (durationUnit == DurationUnit.SECONDS) seconds = Math.round(duration * 1.0);
		else 
		if (durationUnit == DurationUnit.MINUTES) seconds = Math.round(duration * 60.0);
		else 
		if (durationUnit == DurationUnit.HOURS  ) seconds = Math.round(duration * 3600.0);
		
		int tasks = (trainingUnit == TrainingUnit.EPOCHS) ? (ModelConf.getInstance().numberOfTasksPerEpoch() * count) : count;
		
		log.info(String.format("Train for %d seconds or %d tasks", seconds, tasks));
		
		AtomicBoolean finish = new AtomicBoolean (false);
		
		long numberoftasks = 0L;
		
		CountDownTimer timer = new CountDownTimer (finish);
		timer.set(seconds);
		Thread thread = new Thread (timer);
		thread.setName("CountDownTimer");
		thread.start();
		
		while (! finish.get() && numberoftasks < tasks) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			numberoftasks ++;
		}
		
		/* Did we finish because of a time-out? */
		if (finish.get())
			log.info(String.format("Received wake-up call after %d tasks", numberoftasks));
		
		/* Await completion of the last task dispatched. */
		while (dataflows[0].getResultCollector().getNumberOfTasks() < numberoftasks)
			LockSupport.parkNanos(1L);
		
		log.info(String.format("Trained for %d tasks", numberoftasks));
		
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void trainUntilCondition (float target) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid = 0;
		int lowerBound;
		
		log.info(String.format("Train until training loss is %.5f", target));
		
		AtomicBoolean finish = new AtomicBoolean (false);
		
		long numberoftasks = 0L;
		
		dataflows[0].getResultHandler ().setTargetLoss (target, finish);
		
		while (! finish.get()) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			numberoftasks ++;
		}
		
		log.info(String.format("Target loss reached after %d tasks", numberoftasks));
		
		/* Await completion of the last task dispatched. */
		while (dataflows[0].getResultCollector().getNumberOfTasks() < numberoftasks)
			LockSupport.parkNanos(1L);
		
		log.info(String.format("Trained for %d tasks", numberoftasks));
		
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void trainUntilCondition (float target, int count, TrainingUnit unit) {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid = 0;
		int lowerBound;
		
		int tasks = (unit == TrainingUnit.EPOCHS) ? (ModelConf.getInstance().numberOfTasksPerEpoch() * count) : count;
		
		log.info(String.format("Train until training loss is %.5f or %d tasks", target, tasks));
		
		AtomicBoolean finish = new AtomicBoolean (false);
		
		long numberoftasks = 0L;
		
		dataflows[0].getResultHandler ().setTargetLoss (target, finish);
		
		while (! finish.get() && numberoftasks < tasks) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			numberoftasks ++;
		}
		
		/* Did we finish because of target loss was reached? */
		if (finish.get())
			log.info(String.format("Target loss reached after %d tasks", numberoftasks));
		
		/* Await completion of the last task dispatched. */
		while (dataflows[0].getResultCollector().getNumberOfTasks() < numberoftasks)
			LockSupport.parkNanos(1L);
		
		log.info(String.format("Trained for %d tasks", numberoftasks));
		
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void trainUntilConditionOrForDuration (float target, double duration, DurationUnit unit) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid = 0;
		int lowerBound;
		
		long seconds = 0L;
		
		if (unit == DurationUnit.SECONDS) seconds = Math.round(duration * 1.0);
		else 
		if (unit == DurationUnit.MINUTES) seconds = Math.round(duration * 60.0);
		else 
		if (unit == DurationUnit.HOURS  ) seconds = Math.round(duration * 3600.0);
		
		log.info(String.format("Train until training loss is %.5f or %d seconds", target, seconds));
		
		AtomicBoolean finish0 = new AtomicBoolean (false); /* Target loss reached */
		AtomicBoolean finish1 = new AtomicBoolean (false); /* Timer fired */
		
		long numberoftasks = 0L;
		
		dataflows[0].getResultHandler ().setTargetLoss (target, finish0);
		
		CountDownTimer timer = new CountDownTimer (finish1);
		timer.set(seconds);
		Thread thread = new Thread (timer);
		thread.setName("CountDownTimer");
		thread.start();
		
		while (! finish0.get() && ! finish1.get()) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			numberoftasks ++;
		}
		
		/* Did we finish because of target loss was reached? */
		if (finish0.get())
			log.info(String.format("Target loss reached after %d tasks", numberoftasks));
		else
			log.info(String.format("Received wake-up call after %d tasks", numberoftasks));
		
		/* Await completion of the last task dispatched. */
		while (dataflows[0].getResultCollector().getNumberOfTasks() < numberoftasks)
			LockSupport.parkNanos(1L);
		
		log.info(String.format("Trained for %d tasks", numberoftasks));
		
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void trainUntilConditionOrForDuration 
		(float target, double duration, DurationUnit durationUnit, int count, TrainingUnit trainingUnit) throws Exception {
		
		int slack = ModelConf.getInstance().getSlack();
		int [] bound = new int [1];
		
		int taskid = 0;
		int lowerBound;
		
		long seconds = 0;
		
		if (durationUnit == DurationUnit.SECONDS) seconds = Math.round(duration * 1.0);
		else 
		if (durationUnit == DurationUnit.MINUTES) seconds = Math.round(duration * 60.0);
		else 
		if (durationUnit == DurationUnit.HOURS  ) seconds = Math.round(duration * 3600.0);
		
		int tasks = (trainingUnit == TrainingUnit.EPOCHS) ? (ModelConf.getInstance().numberOfTasksPerEpoch() * count) : count;
		
		log.info(String.format("Train until training loss is %.5f or %d seconds or %d tasks", target, seconds, tasks));
		
		AtomicBoolean finish0 = new AtomicBoolean (false); /* Target loss reached */
		AtomicBoolean finish1 = new AtomicBoolean (false); /* Timer fired */
		
		long numberoftasks = 0L;
		
		dataflows[0].getResultHandler ().setTargetLoss (target, finish0);
		
		CountDownTimer timer = new CountDownTimer (finish1);
		timer.set(seconds);
		Thread thread = new Thread (timer);
		thread.setName("CountDownTimer");
		thread.start();
		
		while (! finish0.get() && ! finish1.get() && numberoftasks < tasks) {
			
			taskid = workClock.incrementAndGetNext(bound);
			lowerBound = bound[0] - slack - 1;
			dataflows[0].getTaskDispatcher().dispatchNext(taskid, lowerBound);
			numberoftasks ++;
		}
		
		/* Did we finish because of target loss was reached? */
		if (finish0.get()) {
			log.info(String.format("Target loss reached after %d tasks", numberoftasks));
		} else
		if (finish1.get()) {
			log.info(String.format("Received wake-up call after %d tasks", numberoftasks));
		} else {
			log.info(String.format("Dispatched up to %d tasks", numberoftasks));
		}
		
		/* Await completion of the last task dispatched. */
		while (dataflows[0].getResultCollector().getNumberOfTasks() < numberoftasks)
			LockSupport.parkNanos(1L);
		
		log.info(String.format("Trained for %d tasks", numberoftasks));
		
		/* Flush measurements */
		dataflows[0].getResultHandler().flush();
		return;
	}
	
	public void test () throws Exception {
		
		int tasks = ModelConf.getInstance().numberOfTestTasks();
		int bound = workClock.getClock() - 1;
		/*
		 * Note:
		 * 
		 * If test () is called before a model update occurs,
		 * then the system is dead-locked.
		 */
		
		CountDownLatch latch = new CountDownLatch (tasks);
		dataflows[1].getResultCollector().setCountDownLatch(latch);
		
		for (int id = 1; id <= tasks; ++id) {
			dataflows[1].getTaskDispatcher().dispatchNext(id, bound);
		}
		/* Await task completion */
		latch.await();
	}
	
	public void destroy () throws Exception {
		
		monitor.stop ();
		workerPool.stop();
		
		if(BLAS.getInstance().isLoaded())
			BLAS.getInstance().destroy ();
		
		/* Free dataset file handlers */
		switch (ModelConf.getInstance().getDatasetType()) {
		
		case  BASIC:            DatasetMemoryManager.getInstance().free(); break;
		case  LIGHT: LightWeightDatasetMemoryManager.getInstance().free(); break;
		default:
			break;
		}
		
		TheGPU.getInstance().destroy ();
	}
}
