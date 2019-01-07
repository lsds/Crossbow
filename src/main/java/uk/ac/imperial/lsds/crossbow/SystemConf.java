package uk.ac.imperial.lsds.crossbow;

import java.lang.reflect.Field;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.Map;

import sun.misc.Unsafe;
import uk.ac.imperial.lsds.crossbow.cli.Option;
import uk.ac.imperial.lsds.crossbow.types.ExecutionMode;
import uk.ac.imperial.lsds.crossbow.types.ReplicationModel;
import uk.ac.imperial.lsds.crossbow.types.SchedulingPolicy;
import uk.ac.imperial.lsds.crossbow.types.SynchronisationModel;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

@SuppressWarnings("restriction")
public class SystemConf {
	
	private static final DateFormat sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
	
	private static SystemConf instance = new SystemConf ();
	
	public static SystemConf getInstance () { return instance; }
	
	private static Unsafe theUnsafe;
	
	private static Unsafe getUnsafeMemory () {
		try {
			Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
			theUnsafe.setAccessible(true);
			return (Unsafe) theUnsafe.get (null);
			
		} catch (Exception e) {
			throw new AssertionError(e);
		}
	}
	
	public static String setHomeDirectory () {
		
		Map<String, String> env = System.getenv();
		
		String dir = (String) env.get("CROSSBOW_HOME");
		if (dir == null) {
			System.err.println ("error: CROSSBOW_HOME environment variable is not set");
			System.exit(1);
		}
		return dir;
	}
	
	LinkedList<Option> opts;
	
	private String home;
	
	private String checkpointDirectory, modelDirectory;
	
	private boolean CPU, GPU;
	
	private int [] GPUDevices;
	
	private int workers;
	private int [] replicas;
	
	private int slots;
	
	private SchedulingPolicy schedulingPolicy;
	
	private SynchronisationModel synchronisationModel;
	
	private ReplicationModel replicationModel;
	
	private int readersPerModel;
	
	private int variableBufferSize;
	
	private int buffers;
	
	private boolean directBuffers;
	
	private int streams;
	
	private int callbackhandlers;
	private int taskhandlers;
	private int filehandlers;
	
	private int displayInterval;
	private TrainingUnit displayIntervalUnit;
	private boolean accumulate;
	
	private int checkpointInterval;
	private TrainingUnit checkpointIntervalUnit;
	
	private boolean queue;
	
	private boolean tee;
	
	private long filePartitionSize;
	private int taskQueueSizeLimit;
	
	private long seed;
	
	private boolean reuseMemory;
	
	private int mappings;
	
	private long performanceMonitorInterval;
	
	private CoreMapper mapper;

	/* Auto-tuning configuration parameters */
	private boolean autotune;
	private double  autotuneThreshold;
	private int     autotuneInterval;
	
	/* 
	 * Direct scheduling should be enabled only in
	 * GPU-only mode.
	 * 
	 * When enabled the dispatcher skips the queue
	 * and schedules tasks directly via JNI call.
	 */
	private boolean directScheduling;
	
	private SystemConf () {
		
		theUnsafe = getUnsafeMemory ();
		
		home = setHomeDirectory ();
		
		/* Fill command-line arguments */
		opts = new LinkedList<Option> ();
		
		opts.add (new Option ("--cpu"                        ).setType (Boolean.class));
		opts.add (new Option ("--gpu"                        ).setType (Boolean.class));
		opts.add (new Option ("--gpu-devices"                ).setType ( String.class));
		opts.add (new Option ("--checkpoint-directory"       ).setType ( String.class));
		opts.add (new Option ("--model-directory"            ).setType ( String.class));
		opts.add (new Option ("--number-of-workers"          ).setType (Integer.class));
		opts.add (new Option ("--number-of-cpu-models"       ).setType (Integer.class));
		opts.add (new Option ("--number-of-gpu-models"       ).setType (Integer.class));
		opts.add (new Option ("--number-of-result-slots"     ).setType (Integer.class));
		opts.add (new Option ("--scheduling-policy"          ).setType ( String.class));
		opts.add (new Option ("--synchronisation-model"      ).setType ( String.class));
		opts.add (new Option ("--replication-model"          ).setType ( String.class));
		opts.add (new Option ("--readers-per-model"          ).setType (Integer.class));
		opts.add (new Option ("--data-buffer-size"           ).setType (Integer.class));
		opts.add (new Option ("--number-of-buffers"          ).setType (Integer.class));
		opts.add (new Option ("--use-direct-buffers"         ).setType (Boolean.class));
		opts.add (new Option ("--number-of-gpu-streams"      ).setType (Integer.class));
		opts.add (new Option ("--number-of-callback-handlers").setType (Integer.class));
		opts.add (new Option ("--number-of-task-handlers"    ).setType (Integer.class));
		opts.add (new Option ("--number-of-file-handlers"    ).setType (Integer.class));
		opts.add (new Option ("--display-interval"           ).setType (Integer.class));
		opts.add (new Option ("--display-interval-unit"      ).setType ( String.class));
		opts.add (new Option ("--display-accumulated-loss"   ).setType (Boolean.class));
		opts.add (new Option ("--checkpoint-interval"        ).setType (Integer.class));
		opts.add (new Option ("--checkpoint-interval-unit"   ).setType ( String.class));
		opts.add (new Option ("--queue-measurements"         ).setType (Boolean.class));
		opts.add (new Option ("--file-partition-size"        ).setType (   Long.class));
		opts.add (new Option ("--task-queue-size"            ).setType (Integer.class));
		opts.add (new Option ("--random-seed"                ).setType (   Long.class));
		opts.add (new Option ("--reuse-memory"               ).setType (Boolean.class));
		opts.add (new Option ("--mapped-partitions-window"   ).setType (Integer.class));
		opts.add (new Option ("--monitor-interval"           ).setType (   Long.class));
		opts.add (new Option ("--tee-measurements"           ).setType (Boolean.class));
		opts.add (new Option ("--autotune-models"            ).setType (Boolean.class));
		opts.add (new Option ("--autotune-threshold"         ).setType ( Double.class));
		opts.add (new Option ("--autotune-interval"          ).setType (Integer.class));
		opts.add (new Option ("--direct-scheduling"          ).setType (Boolean.class));
		
		/* Default values */
		
		CPU = true;
		GPU = false;
		
		GPUDevices = new int [] { 0 }; /* Use first available device */
		
		checkpointDirectory = modelDirectory = null;
		
		workers = 1;
		replicas = new int [2];
		replicas[0] = replicas[1] = 1;
		
		slots = 2048;
		
		schedulingPolicy = SchedulingPolicy.FIFO;
		synchronisationModel = SynchronisationModel.BSP;
		
		replicationModel = ReplicationModel.SRSW;
		readersPerModel = 1;
		
		variableBufferSize = 1048576;
		
		buffers = 256;
		
		directBuffers = true;
		
		streams = 1;
		
		callbackhandlers = 1;
		taskhandlers = 1;
		filehandlers = 1;
		
		displayInterval = 1;
		displayIntervalUnit = TrainingUnit.TASKS;
		accumulate = false;
		
		checkpointInterval = 0;
		checkpointIntervalUnit = TrainingUnit.TASKS;
		
		queue = false;
		
		filePartitionSize = 1073741824L;
		
		taskQueueSizeLimit = 32;
		
		seed = 123456789L;
		
		reuseMemory = false;
		
		mappings = 4;
		
		performanceMonitorInterval = 1000L;
		
		tee = true;
		
		autotune = false;
		
		autotuneThreshold = 0.1D;
		autotuneInterval  = 1;
		
		directScheduling = false;
		
		mapper = new CoreMapper ();
	}
	
	public String getHomeDirectory () {
		return home;
	}
	
	public SystemConf setCheckpointDirectory (String checkpointDirectory) {
		this.checkpointDirectory = checkpointDirectory;
		return this;
	}
	
	public String getCheckpointDirectory () {
		return checkpointDirectory;
	}
	
	public SystemConf setModelDirectory (String modelDirectory) {
		this.modelDirectory = modelDirectory;
		return this;
	}
	
	public String getModelDirectory () {
		return modelDirectory;
	}
	
	public SystemConf setCPU (boolean CPU) {
		this.CPU = CPU;
		return this;
	}
	
	public SystemConf setGPU (boolean GPU) {
		this.GPU = GPU;
		return this;
	}
	
	public boolean getCPU () {
		return CPU;
	}
	
	public boolean getGPU () {
		return GPU;
	}
	
	public ExecutionMode getExecutionMode () {
		if (CPU && (! GPU)) return ExecutionMode.CPU;
		else
		if ((! CPU) && GPU) return ExecutionMode.GPU;
		else
			return ExecutionMode.HYBRID;
	}
	
	public boolean isHybrid () {
		return (getExecutionMode() == ExecutionMode.HYBRID);
	}
	
	public SystemConf setNumberOfWorkerThreads (int workers) {
		this.workers = workers;
		return this;
	}
	
	public int numberOfWorkerThreads () {
		return workers;
	}
	
	public SystemConf setNumberOfCPUModelReplicas (int replicas) {
		this.replicas[0] = replicas;
		return this;
	}
	
	public int numberOfCPUModelReplicas () {
		return replicas[0];
	}
	
	public SystemConf setNumberOfGPUModelReplicas (int replicas) {
		this.replicas[1] = replicas;
		return this;
	}
	
	public int numberOfGPUModelReplicas () {
		return replicas[1];
	}
	
	public SystemConf setNumberOfResultSlots (int slots) {
		this.slots = slots;
		return this;
	}
	
	public int numberOfResultSlots () {
		return slots;
	}
	
	public SystemConf setSchedulingPolicy (SchedulingPolicy schedulingPolicy) {
		this.schedulingPolicy = schedulingPolicy;
		return this;
	}
	
	public SchedulingPolicy getSchedulingPolicy () {
		return schedulingPolicy;
	}
	
	public SystemConf setSynchronisationModel (SynchronisationModel synchronisationModel) {
		this.synchronisationModel = synchronisationModel;
		return this;
	}
	
	public SynchronisationModel getSynchronisationModel () {
		return synchronisationModel;
	}
	
	public SystemConf setReplicationModel (ReplicationModel replicationModel) {
		this.replicationModel = replicationModel;
		return this;
	}
	
	public ReplicationModel getReplicationModel () {
		return replicationModel;
	}
	
	public SystemConf setNumberOfReadersPerModel (int readersPerModel) {
		this.readersPerModel = readersPerModel;
		return this;
	}
	
	public int numberOfReadersPerModel () {
		return readersPerModel;
	}
	
	public SystemConf setNumberOfBuffers (int buffers) {
		this.buffers = buffers;
		return this;
	}
	
	public int numberOfBuffers () {
		return buffers;
	}
	
	public SystemConf setVariableBufferSize (int variableBufferSize) {
		this.variableBufferSize = variableBufferSize;
		return this;
	}
	
	public int getVariableBufferSize () {
		return variableBufferSize;
	}
	
	public SystemConf useDirectBuffers (boolean direct) {
		this.directBuffers = direct;
		return this;
	}
	
	public boolean useDirectBuffers () {
		return directBuffers;
	}
	
	public SystemConf setNumberOfGPUStreams (int streams) {
		this.streams = streams;
		return this;
	}
	
	public int numberOfGPUStreams () {
		return streams;
	}
	
	public SystemConf setNumberOfGPUCallbackHandlers (int callbackhandlers) {
		this.callbackhandlers = callbackhandlers;
		return this;
	}
	
	public int numberOfGPUCallbackHandlers () {
		return callbackhandlers;
	}
	
	public SystemConf setNumberOfGPUTaskHandlers (int taskhandlers) {
		this.taskhandlers = taskhandlers;
		return this;
	}
	
	public int numberOfGPUTaskHandlers () {
		return taskhandlers;
	}
	
	public SystemConf setNumberOfFileHandlers (int filehandlers) {
		this.filehandlers = filehandlers;
		return this;
	}
	
	public int numberOfFileHandlers () {
		return filehandlers;
	}
	
	public SystemConf setDisplayInterval (int displayInterval) {
		this.displayInterval = displayInterval;
		return this;
	}
	
	public int getDisplayInterval () {
		return ((displayIntervalUnit == TrainingUnit.EPOCHS) ? (displayInterval * ModelConf.getInstance().numberOfTasksPerEpoch ()) : displayInterval);
	}
	
	public SystemConf setDisplayIntervalUnit (TrainingUnit displayIntervalUnit) {
		this.displayIntervalUnit = displayIntervalUnit;
		return this;
	}
	
	public TrainingUnit getDisplayIntervalUnit () {
		return displayIntervalUnit;
	}
	
	public SystemConf displayAccumulatedLossValue (boolean accumulate) {
		this.accumulate = accumulate;
		return this;
	}
	
	public boolean displayAccumulatedLossValue () {
		return accumulate;
	}
	
	public SystemConf setCheckpointInterval (int checkpointInterval) {
		this.checkpointInterval = checkpointInterval;
		return this;
	}
	
	public int getCheckpointInterval () {
		return ((checkpointIntervalUnit == TrainingUnit.EPOCHS) ? (checkpointInterval * ModelConf.getInstance().numberOfTasksPerEpoch ()) : checkpointInterval);
	}
	
	public SystemConf setCheckpointIntervalUnit (TrainingUnit checkpointIntervalUnit) {
		this.checkpointIntervalUnit = checkpointIntervalUnit;
		return this;
	}
	
	public TrainingUnit getCheckpointIntervalUnit () {
		return checkpointIntervalUnit;
	}
	
	public SystemConf queueMeasurements (boolean queue) {
		this.queue = queue;
		return this;
	}
	
	public boolean queueMeasurements () {
		return queue;
	}
	
	public int getPageSize () {
		return theUnsafe.pageSize();
	}
	
	public SystemConf setFilePartitionSize (long filePartitionSize) {
		this.filePartitionSize = filePartitionSize;
		return this;
	}
	
	public long getFilePartitionSize () {
		return filePartitionSize;
	}
	
	public SystemConf setTaskQueueSizeLimit (int taskQueueSizeLimit) {
		
		this.taskQueueSizeLimit = taskQueueSizeLimit;
		return this;
	}
	
	public int getTaskQueueSizeLimit () {
		
		return taskQueueSizeLimit;
	}
	
	public SystemConf setRandomSeed (long seed) {
		this.seed = seed;
		return this;
	}
	
	public long getRandomSeed () {
		return seed;
	}
	
	public SystemConf allowMemoryReuse (boolean reuseMemory) {
		this.reuseMemory = reuseMemory;
		return this;
	}
	
	public boolean tryReuseMemory () {
		return reuseMemory;
	}
	
	public SystemConf setTeeMeasurements (boolean tee) {
		this.tee = tee;
		return this;
	}
	
	public boolean teeMeasurements () {
		return tee;
	}
	
	public SystemConf setMaxNumberOfMappedPartitions(int mappings) {
		this.mappings = mappings;
		return this;
	}
	
	public int maxNumberOfMappedPartitions() {
		return mappings;
	}
	
	public SystemConf setPerformanceMonitorInterval (long performanceMonitorInterval) {
		this.performanceMonitorInterval = performanceMonitorInterval;
		return this;
	}
	
	public long getPerformanceMonitorInterval() {
		return performanceMonitorInterval;
	}
	
	public SystemConf autotuneModels (boolean autotune) {
		this.autotune = autotune;
		return this;
	}
	
	public boolean autotuneModels () {
		return autotune;
	}
	
	public SystemConf setAutotuneInterval (int autotuneInterval) {
		this.autotuneInterval = autotuneInterval;
		return this;
	}
	
	public int getAutotuneInterval () {
		return autotuneInterval;
	}
	
	public SystemConf setAutotuneThreshold (double autotuneThreshold) {
		this.autotuneThreshold = autotuneThreshold;
		return this;
	}
	
	public double getAutotuneThreshold () {
		return autotuneThreshold;
	}
	
	public SystemConf useDirectScheduling (boolean directScheduling) {
		this.directScheduling = directScheduling;
		return this;
	}
	
	public boolean useDirectScheduling () {
		return directScheduling;
	}
	
	public SystemConf setGPUDevices (int ... GPUDevices) {
		this.GPUDevices = GPUDevices;
		return this;
	}
	
	public int [] getGPUDevices () {
		return GPUDevices;
	}
	
	public CoreMapper getCoreMapper () {
		return mapper;
	}
	
	public boolean parse (String arg, Option opt) {
		
		if (arg.equals("--cpu")) {
			
			setCPU (opt.getBooleanValue ());
		}
		else if (arg.equals("--gpu")) {
			
			setGPU (opt.getBooleanValue ());
		}
		else if (arg.equals("--gpu-devices")) {
			
			String [] tokens = opt.getStringValue ().split (",");
			if (tokens.length < 1) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
			
			int [] list = new int [tokens.length];
			try {
				for (int i = 0; i < list.length; i++)
					list[i] = Integer.parseInt(tokens[i]);
			}
			catch (NumberFormatException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
			
			setGPUDevices (list);
		}
		else if (arg.equals("--checkpoint-directory")) {
			
			setCheckpointDirectory (opt.getStringValue ());
		}
		else if (arg.equals("--model-directory")) {
			
			setModelDirectory (opt.getStringValue ());
		}
		else if (arg.equals("--number-of-workers")) {
			
			setNumberOfWorkerThreads (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-cpu-models")) {
			
			setNumberOfCPUModelReplicas (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-gpu-models")) {
			
			setNumberOfGPUModelReplicas (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-result-slots")) {
			
			setNumberOfResultSlots (opt.getIntValue ());
		} 
		else if (arg.equals("--scheduling-policy")) {
			
			try {
				setSchedulingPolicy (SchedulingPolicy.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		} 
		else if (arg.equals("--synchronisation-model")) {
			
			try {
				setSynchronisationModel (SynchronisationModel.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		} 
		else if (arg.equals("--replication-model")) {
			
			try {
				setReplicationModel (ReplicationModel.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		} 
		else if (arg.equals("--readers-per-model")) {
			
			setNumberOfReadersPerModel (opt.getIntValue ());
		} 
		else if (arg.equals("--data-buffer-size")) {
			
			setVariableBufferSize (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-buffers")) {
			
			setNumberOfBuffers (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-gpu-streams")) {
			
			setNumberOfGPUStreams (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-callback-handlers")) {
			
			setNumberOfGPUCallbackHandlers (opt.getIntValue ());
		} 
		else if (arg.equals("--number-of-task-handlers")) {
			
			setNumberOfGPUTaskHandlers (opt.getIntValue ());
		}
		else if (arg.equals("--number-of-file-handlers")) {
			
			setNumberOfFileHandlers (opt.getIntValue ());
		}
		else if (arg.equals("--display-interval")) {
			
			setDisplayInterval (opt.getIntValue ());
		}
		else if (arg.equals("--display-interval-unit")) {
			
			try {
				setDisplayIntervalUnit (TrainingUnit.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--display-accumulated-loss")) {
			
			displayAccumulatedLossValue (opt.getBooleanValue ());
		}
		else if (arg.equals("--checkpoint-interval")) {
			
			setCheckpointInterval (opt.getIntValue ());
		}
		else if (arg.equals("--checkpoint-interval-unit")) {
			
			try {
				setCheckpointIntervalUnit (TrainingUnit.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--queue-measurements")) {
			
			queueMeasurements (opt.getBooleanValue ());
		} 
		else if (arg.equals("--file-partition-size")) {
			
			setFilePartitionSize (opt.getLongValue ());
		} 
		else if (arg.equals("--task-queue-size")) {
			
			setTaskQueueSizeLimit (opt.getIntValue ());
		}
		else if (arg.equals("--reuse-memory")) {
			
			allowMemoryReuse (opt.getBooleanValue ());
		} 
		else if (arg.equals("--mapped-partitions-window")) {
			
			setMaxNumberOfMappedPartitions (opt.getIntValue ());
		}
		else if (arg.equals("--monitor-interval")) {
			
			setPerformanceMonitorInterval (opt.getLongValue ());
		}
		else if (arg.equals("--autotune-models")) {
			
			autotuneModels (opt.getBooleanValue ());
		}
		else if (arg.equals("--autotune-threshold")) {
			
			setAutotuneThreshold (opt.getDoubleValue ());
		}
		else if (arg.equals("--autotune-interval")) {
			
			setAutotuneInterval (opt.getIntValue ());
		}
		else if (arg.equals("--direct-scheduling")) {
			
			useDirectScheduling (opt.getBooleanValue ());
		}
		else {
			return false;
		}
		/* Valid system configuration option */
		return true;
	}
	
	public void dump () {
		
		StringBuilder s = new StringBuilder (String.format("=== [System configuration (%s)] ===\n", sdf.format(new Date())));
		
		s.append(String.format("Home directory is %s\n", home));
		s.append(String.format("Checkpoint directory is %s\n", checkpointDirectory));
		s.append(String.format("%s execution mode\n", (isHybrid() ? "Hybrid" : (getGPU() ? "GPU-only" : "CPU-only"))));
		s.append(String.format("%d worker threads\n", workers));
		s.append(String.format("%d CPU model replicas\n", replicas[0]));
		s.append(String.format("%d reader%s per CPU model replica\n", readersPerModel, (readersPerModel > 1 ? "s" : "")));
		s.append(String.format("%d GPU model replicas\n", replicas[1]));
		s.append(String.format("%d GPU streams\n", streams));
		s.append(String.format("%d GPU task callback handlers\n", callbackhandlers));
		s.append(String.format("%d GPU task handlers\n", taskhandlers));
		s.append(String.format("%d dataset file handlers\n", filehandlers));
		s.append(String.format("%d result slots\n", slots));
		s.append(String.format("Scheduling policy is %s\n", schedulingPolicy.toString()));
		s.append(String.format("Synchronisation model is %s\n", synchronisationModel.toString()));
		s.append(String.format("Replication model is %s\n", replicationModel.toString()));
		s.append(String.format("%d buffers in BLAS buffer pool\n", buffers));
		s.append(String.format("%d bytes per BLAS buffer\n", variableBufferSize));
		s.append(String.format("%s direct buffers\n", (directBuffers ? "Use" : "Don't use")));
		s.append(String.format("Display loss every %d tasks\n", getDisplayInterval()));
		s.append(String.format("%s loss measurements\n", (accumulate ? "Average" : "Don't average")));
		s.append(String.format("%s loss measurements\n", (queue ? "Queue" : "Don't queue")));
		s.append(String.format("%s measurements\n", (tee ? "Tee" : "Don't tee")));
		s.append(String.format("%d bytes per input data partition [max]\n", filePartitionSize));
		s.append(String.format("%d tasks queued [max]\n", taskQueueSizeLimit));
		s.append(String.format("Random seed is %d\n", seed));
		s.append(String.format("Performance monitor interval is %d\n", performanceMonitorInterval));
		s.append(String.format("%s number of model replicas per GPU\n", (autotune ? "Auto-tune" : "Don't auto-tune")));
		if (autotune) {
			s.append(String.format("Auto-tune number of model replicas every %d synchronisation cycles\n", autotuneInterval));
			s.append(String.format("Auto-tuning throughput improvement threshold is %5.3f\n", autotuneThreshold));
		}
		s.append(String.format("%s direct scheduling\n", (directScheduling ? "Use" : "Don't use")));
		
		s.append("=== [End of system configuration dump] ===");
		
		/* Dump on screen */
		System.out.println(s.toString());
	}
	
	public LinkedList<Option> getOptions () {
		return opts;
	}
}
