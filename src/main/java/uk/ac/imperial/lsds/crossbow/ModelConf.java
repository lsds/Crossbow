package uk.ac.imperial.lsds.crossbow;

import java.util.LinkedList;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.Option;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.dataset.DatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.device.dataset.LightWeightDatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;
import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Regularisation;
import uk.ac.imperial.lsds.crossbow.types.MomentumMethod;

public class ModelConf {
	
	private final static Logger log = LogManager.getLogger (ModelConf.class);
	
	private static final ModelConf instance = new ModelConf ();
	
	public static ModelConf getInstance () { return instance; }
	
	LinkedList<Option> opts;
	
	private TrainingUnit wpcUnit, testIntervalUnit;
	
	private IDataset [] datasets;
	
	private int wpc;
	private int slack;
	
	private int batchSize;
	private int splits;
	
	private int testInterval;
	
	private UpdateModel updateModel;
	
	private SolverConf solverConf;
	
	public ModelConf () {
		
		/* Fill command-line arguments */
		opts = new LinkedList<Option>();
		
		opts.add (new Option ("--test-interval-unit").setType ( String.class));
		opts.add (new Option ("--wpc-unit"          ).setType ( String.class));
		opts.add (new Option ("--wpc"               ).setType (Integer.class));
		opts.add (new Option ("--slack"             ).setType (Integer.class));
		opts.add (new Option ("--batch-size"        ).setType (Integer.class));
		opts.add (new Option ("--splits"            ).setType (Integer.class));
		opts.add (new Option ("--test-interval"     ).setType (Integer.class));
		opts.add (new Option ("--update-model"      ).setType ( String.class));
		
		/* Add solver command-line arguments */
		opts.addAll(SolverConf.getOptions ());
		
		/* Default values */
		
		wpcUnit = TrainingUnit.TASKS;
		testIntervalUnit = TrainingUnit.TASKS;
		
		datasets = new IDataset [2];
		datasets[0] = datasets[1] = null;
		
		wpc = 1;
		slack = 0;
		
		batchSize = 32;
		splits = 1;
		
		testInterval = 1;
		
		updateModel = UpdateModel.DEFAULT;
		
		solverConf = new SolverConf (this);
	}
	
	public ModelConf setWpcUnit (TrainingUnit wpcUnit) {
		this.wpcUnit = wpcUnit;
		return this;
	}
	
	public TrainingUnit getWpcUnit () {
		return wpcUnit;
	}
	
	public ModelConf setTestIntervalUnit (TrainingUnit testIntervalUnit) {
		this.testIntervalUnit = testIntervalUnit;
		return this;
	}
	
	public TrainingUnit getTestIntervalUnit () {
		return testIntervalUnit;
	}
	
	public ModelConf setDataset (Phase phase, IDataset dataset) {
		if (dataset != null)
			dataset.setPhase (phase);
		int id = phase.getId();
		datasets[id] = dataset;
		return this;
	}
	
	public ModelConf setTrainingDataset (IDataset dataset) {
		return setDataset (Phase.TRAIN, dataset);
	}
	
	public ModelConf setValidationDataset (IDataset dataset) {
		return setDataset (Phase.CHECK, dataset);
	}
	
	public IDataset getDataset (Phase phase) {
		int id = phase.getId();
		if (! datasets[id].isInitialised())
			throw new IllegalStateException (String.format("Dataset for %s phase is not initialised", phase.toString()));
		return datasets[id];
	}
	
	public IDataset getTrainingDataset () {
		return getDataset (Phase.TRAIN);
	}
	
	public IDataset getValidationDataset () {
		return getDataset (Phase.CHECK);
	}
	
	public ModelConf setWpc (int wpc) {
		this.wpc = wpc;
		return this;
	}
	
	public int getWpc () {
		/* Always return `wpc` in terms of tasks. If unit is `EPOCHS`, convert it */
		return ((wpcUnit == TrainingUnit.EPOCHS) ? (wpc * numberOfTasksPerEpoch ()) : wpc);
	}
	
	public ModelConf setSlack (int slack) {
		this.slack = slack;
		return this;
	}
	
	/*
	 * Slack is always expressed as a function of a clock tick (`wpc`).
	 */
	public int getSlack () {
		return slack;
	}
	
	public ModelConf setBatchSize (int microBatchSize) {
		this.batchSize = microBatchSize;
		return this;
	}
	
	public int getBatchSize () {
		return batchSize;
	}
	
	public ModelConf setNumberOfSplits (int splits) {
		this.splits = splits;
		return this;
	}
	
	public int numberOfSplits () {
		return splits;
	}
	
	public ModelConf setTestInterval (int testInterval) {
		this.testInterval = testInterval;
		return this;
	}

	public int getTestInterval () {
		/* Always return `testInverval` in terms of tasks. If unit is `EPOCHS`, convert it */
		return ((testIntervalUnit == TrainingUnit.EPOCHS) ? (testInterval * numberOfTasksPerEpoch ()) : testInterval);
	}
	
	public ModelConf setUpdateModel (UpdateModel updateModel) {
		this.updateModel = updateModel;
		return this;
	}
	
	public UpdateModel getUpdateModel () {
		return updateModel;
	}
	
	public ModelConf setSolverConf (SolverConf solverConf) {
		this.solverConf = solverConf.setModelConf (this);
		return this;
	}
	
	public SolverConf getSolverConf () {
		return solverConf;
	}
	
	public Shape getInputShape (Phase phase) {
		
		Shape exampleshape = getDataset(phase).getMetadata().getExampleShape();
		
		Shape batchshape = new Shape (exampleshape.dimensions() + 1);
		
		/* Assumes that init() has been called */
		if (splits > 1) {
			batchshape.set(0, batchSize / splits);
		} else {
			batchshape.set(0, batchSize);
		}
		
		for (int i = 0; i < exampleshape.dimensions(); ++i)
			batchshape.set(i + 1, exampleshape.get(i));
		
		return batchshape;
	}
	
	public int numberOfTasksPerEpoch () {
		
		int count = getDataset(Phase.TRAIN).getMetadata().numberOfExamples();
		
		if ((count % batchSize) != 0)
			throw new IllegalStateException ("error: number of training examples is not a multiple of the batch size");
		
		return (count / batchSize);
	}
	
	public int numberOfTestTasks () {
		
		int count = getDataset(Phase.CHECK).getMetadata().numberOfExamples();
		
		if ((count % batchSize) != 0)
			throw new IllegalStateException ("error: number of test examples is not a multiple of the batch size");
		
		return (count / batchSize);
	}
	
	public int [] getTaskSize () {
		
		DatasetMetadata meta = getDataset(Phase.TRAIN).getMetadata();
		
		return new int [] { 
			
			meta.getExampleSize() * batchSize + meta.getExamplesFilePad(), 
			meta.getLabelSize()   * batchSize + meta.getLabelsFilePad()
		};
	}
	
	public boolean parse (String arg, Option opt) {
		
		if (arg.equals("--test-interval-unit")) {
			
			try {
				setTestIntervalUnit (TrainingUnit.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		} 
		else if (arg.equals("--wpc-unit")) {
			
			try {
				setWpcUnit (TrainingUnit.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		} 
		else if (arg.equals("--wpc")) {
			
			setWpc (opt.getIntValue ());
		}
		else if (arg.equals("--slack")) {
			
			setSlack (opt.getIntValue ());
		}
		else if (arg.equals("--batch-size")) {
			
			setBatchSize (opt.getIntValue ());
		}
		else if (arg.equals("--splits")) {
			
			setNumberOfSplits (opt.getIntValue ());
		}
		else if (arg.equals("--test-interval")) {
			
			setTestInterval (opt.getIntValue ());
		}
		else if (arg.equals("--update-model")) {
			
			try {
				setUpdateModel(UpdateModel.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--learning-rate-decay-policy")) {
			
			try {
				solverConf.setLearningRateDecayPolicy (LearningRateDecayPolicy.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--learning-rate")) {
			
			solverConf.setBaseLearningRate (opt.getFloatValue ());
		}
		else if (arg.equals("--gamma")) {
			
			solverConf.setGamma (opt.getFloatValue ());
		}
		else if (arg.equals("--power")) {
			
			solverConf.setPower (opt.getFloatValue ());
		}
		else if (arg.equals("--momentum")) {
			
			solverConf.setMomentum (opt.getFloatValue ());
		}
		else if (arg.equals("--momentum-method")) {
			
			try {
				solverConf.setMomentumMethod (MomentumMethod.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--weight-decay")) {
			
			solverConf.setWeightDecay (opt.getFloatValue ());
		}
		else if (arg.equals("--regularisation-type")) {
			
			try {
				solverConf.setRegularisationType (Regularisation.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--clip-gradient-threshold")) {
			
			solverConf.setClipGradientThreshold (opt.getFloatValue ());
		}
		else if (arg.equals("--step-size")) {
			
			solverConf.setStepSize (opt.getIntValue ());
		}
		else if (arg.equals("--step-values")) {
			
			try {
				String value = opt.getStringValue ();
				String [] values = value.split (",");
				
				int [] steps = new int [values.length];
				
				for (int i = 0; i < steps.length; ++i)
					steps [i] = Integer.parseInt (values[i]); 
				
				solverConf.setStepValues(steps);
				
			}
			catch (Exception e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--learning-rate-step-unit")) {
			
			try {
				solverConf.setLearningRateStepUnit (TrainingUnit.fromString (opt.getStringValue ()));
			}
			catch (IllegalArgumentException e) {
				System.err.println(String.format("error: invalid option: %s %s", arg, opt.getStringValue ()));
				System.exit(1);
			}
		}
		else if (arg.equals("--warmup-steps")) {

			solverConf.setWarmupSteps (opt.getIntValue ());
		}
		else if (arg.equals("--alpha")) {
			
			solverConf.setAlpha (opt.getFloatValue ());
		}
		else if (arg.equals("--tau")) {
			
			solverConf.setTau (opt.getIntValue ());
		}
		else if (arg.equals("--base-model-momentum")) {
			
			solverConf.setBaseModelMomentum (opt.getFloatValue ());
		}
		else if (arg.equals("--min-learning-rate")) {
			
			solverConf.setMinLearningRate (opt.getFloatValue ());
		}
		else if (arg.equals("--max-learning-rate")) {
			
			solverConf.setMaxLearningRate (opt.getFloatValue ());
		}
		else if (arg.equals("--learning-rate-increment")) {
			
			solverConf.setLearningRateIncrement (opt.getFloatValue ());
		}
		else if (arg.equals("--super-convergence")) {
			
			solverConf.setSuperConvergence (opt.getBooleanValue ());
		}
		else if (arg.equals("--min-momentum")) {
			
			solverConf.setMinMomentum (opt.getFloatValue ());
		}
		else if (arg.equals("--max-momentum")) {
			
			solverConf.setMaxMomentum (opt.getFloatValue ());
		}
		else if (arg.equals("--momentum-increment")) {
		
			solverConf.setMomentumIncrement (opt.getFloatValue ());
		}
		else {
			return false;
		}
		/* Valid model configuration option */
		return true;
	}
	
	public void init () {
		
		/* Check number of splits */
		if (splits > 1) {
			if (splits > batchSize) {
				System.err.println(String.format("error: cannot split %d items into %d parts", batchSize, splits));
				System.exit(1);
			}
			if ((batchSize % splits) != 0) {
				System.err.println(String.format("error: batch size %d is not divisible by %d", batchSize, splits));
				System.exit(1);
			}
		}
		
		/* Load the dataset library */
		switch (getDatasetType()) {
		case  BASIC:
			DatasetMemoryManager.getInstance().init ();
			for (int i = 0; i < datasets.length; ++i) {
				if (datasets[i] != null) {
					log.info(String.format("[DBG] Initialise dataset #%d", i));
					datasets[i].init();
				}
			}
			break;
		case  LIGHT:
			LightWeightDatasetMemoryManager.getInstance().init ();
			for (int i = 0; i < datasets.length; ++i) {
				if (datasets[i] != null) {
					log.info(String.format("[DBG] Initialise dataset #%d", i));
					datasets[i].init();
					/* 
					 * Configure the light-weight dataset slot handler 
					 * in the main execution context. 
					 */
					TheGPU.getInstance().setLightWeightDatasetHandler (i, datasets[i].getDatasetSlots(), datasets[i].numberOfSlots());
				}
			}
			break;
		case RECORD:
			for (int i = 0; i < datasets.length; ++i) {
				if (datasets[i] != null) {
					log.info(String.format("[DBG] Initialise dataset #%d", i));
					datasets[i].init();
				}
			}
			break;
		}
		
		int [] tasksize = getTaskSize ();
		
		System.out.println(String.format("[DBG] %d examples/task; %d tasks/epoch; %d and %d bytes/task for examples and labels respectively", 
			getBatchSize(), 
			numberOfTasksPerEpoch(),
			tasksize[0], 
			tasksize[1]
			));
	}
	
	public void dump () {
		
		if (datasets [0] == null) {
			System.err.println("error: dataset is not set");
			System.exit(1);
		}
		
		StringBuilder s = new StringBuilder ("=== [Model configuration dump] ===\n");
		
		s.append (String.format("Synchronise every %d tasks with slack %d\n", getWpc (), slack));
		s.append (String.format("%d examples per batch\n", batchSize));
		s.append (String.format("%d split%s\n", splits, ((splits > 1) ? "s" : "")));
		s.append (String.format("Test every %d tasks\n", getTestInterval ()));
		s.append (String.format("Synchronise with %s\n", getUpdateModel ()));
		s.append (String.format("Using %s datasets\n", getDatasetType().toString()));
		
		/* Append solver configuration */
		s.append("===\n");
		s.append(solverConf.toString());
		s.append("===\n");
		s.append("=== [End of model configuration dump] ===\n");
		
		/* Dump on screen */
		System.out.println(s.toString());
	}

	public LinkedList<Option> getOptions () {
		return opts;
	}
	
	public DatasetType getDatasetType () {
		for (int i = 0; i < datasets.length; ++i)
			if (datasets[i] != null)
				return datasets[i].getType();
		return DatasetType.BASIC;
	}

	public boolean isLightWeight () {
		
		return (getDatasetType() == DatasetType.LIGHT);
	}
}
