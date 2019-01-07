package uk.ac.imperial.lsds.crossbow.kernel.conf;

import java.util.LinkedList;

import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.cli.Option;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.MomentumMethod;
import uk.ac.imperial.lsds.crossbow.types.Regularisation;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class SolverConf implements IConf {
	
	private LearningRateDecayPolicy learningRateDecayPolicy;
	
	private float baseLearningRate;
	
	private float gamma;
	private float power;
	
	private float momentum;
	private MomentumMethod momentumMethod;
	
	private float weightDecay;
	
	private Regularisation regularizationType;
	private float clipGradientThreshold;
	
	private int stepsize;
	
	private int currentstep;
	private int [] stepvalues;
	
	private TrainingUnit stepUnit;
	private boolean converted;
	
	private float alpha;
	private int tau;
	
	/* For EAMSGD */
	private float baseModelMomentum;
	
	/* For super-convergence */
	private float minLearningRate;
	private float maxLearningRate;
	private float incLearningRate;
	
	private boolean superConvergence;
	
	private float minMomentum;
	private float maxMomentum;
	private float incMomentum;
	
	private ModelConf parent;
	
	public SolverConf () {
		this (null);
	}
	
	public SolverConf (ModelConf parent) {
		
		this.parent = parent;
		
		learningRateDecayPolicy = LearningRateDecayPolicy.INV;
		
		baseLearningRate = 0F;
		
		gamma = 0F;
		power = 0F;
		
		momentum = 0F;
		momentumMethod = MomentumMethod.POLYAK;
		
		weightDecay = 0F;
		
		regularizationType = Regularisation.L2;
		clipGradientThreshold = -1F;
		
		stepsize = 0;
		
		currentstep = 0;
		stepvalues = null;
		stepUnit = TrainingUnit.TASKS;
		converted = false;
		
		alpha = 0.5F;
		tau = 1;
		
		/* For EAMSGD */
		baseModelMomentum = 0F;
		
		/* For super-convergence */
		minLearningRate = 0F;
		maxLearningRate = 0F;
		incLearningRate = 0F;
		
		superConvergence = false;
		
		minMomentum = 0F;
		maxMomentum = 0F;
		incMomentum = 0F;
	}
	
	public SolverConf setModelConf (ModelConf parent) {
		this.parent = parent;
		return null;
	}
	
	public float getBaseLearningRate () {
		return baseLearningRate;
	}
	public SolverConf setBaseLearningRate (float baseLearningRate) {
		this.baseLearningRate = baseLearningRate;
		return this;
	}

	public LearningRateDecayPolicy getLearningRateDecayPolicy () {
		return learningRateDecayPolicy;
	}
	
	public SolverConf setLearningRateDecayPolicy 
		(LearningRateDecayPolicy learningRateDecayPolicy) {
		
		this.learningRateDecayPolicy = learningRateDecayPolicy;
		return this;
	}

	public float getGamma () {
		return gamma;
	}

	public SolverConf setGamma (float gamma) {
		this.gamma = gamma;
		return this;
	}

	public float getPower() {
		return power;
	}

	public SolverConf setPower (float power) {
		this.power = power;
		return this;
	}

	public float getMomentum () {
		return momentum;
	}

	public SolverConf setMomentum (float momentum) {
		this.momentum = momentum;
		return this;
	}
	
	public MomentumMethod getMomentumMethod () {
		return momentumMethod;
	}

	public SolverConf setMomentumMethod (MomentumMethod momentumMethod) {
		this.momentumMethod = momentumMethod;
		return this;
	}

	public float getWeightDecay () {
		return weightDecay;
	}

	public SolverConf setWeightDecay (float weightDecay) {
		this.weightDecay = weightDecay;
		return this;
	}

	public Regularisation getRegularisationType () {
		return regularizationType;
	}

	public SolverConf setRegularisationType (Regularisation regularizationType) {
		this.regularizationType = regularizationType;
		return this;
	}
	
	public float getClipGradientThreshold () {
		return clipGradientThreshold;
	}
	
	public SolverConf setClipGradientThreshold (float clipGradientThreshold) {
		this.clipGradientThreshold = clipGradientThreshold;
		return this;
	}
	
	public int getStepSize () {
		return stepsize;
	}

	public SolverConf setStepSize (int stepsize) {
		this.stepsize = stepsize;
		return this;
	}
	
	public int getCurrentStep () {
		return currentstep;
	}

	public SolverConf setCurrentStep (int currentstep) {
		this.currentstep = currentstep;
		return this;
	}
	
	public int [] getStepValues () {
		if (converted)
			return stepvalues;
		
		if (stepUnit != TrainingUnit.TASKS) {
			for (int i = 0; i < stepvalues.length; ++i)
				stepvalues[i] *= parent.numberOfTasksPerEpoch();
		}
		converted = true;
		return stepvalues;
	}
	
	public String getStepValuesToString () {
		if (stepvalues == null)
			return null;
		int [] values_ = getStepValues ();
		StringBuilder s = new StringBuilder ();
		for (int i = 0; i < values_.length; ++i) {
			s.append(String.format("%d", values_[i]));
			if (i != (values_.length - 1))
				s.append(", ");
		}
		return s.toString();
	}
	
	public SolverConf setStepValues (int... stepvalues) {
		this.stepvalues = stepvalues;
		return this;
	}
	
	public SolverConf setLearningRateStepUnit (TrainingUnit stepUnit) {
		this.stepUnit = stepUnit;
		return this;
	}
	
	public TrainingUnit getLearningRateStepUnit () {
		return this.stepUnit;
	}
	
	public SolverConf setAlpha (float alpha) {
		this.alpha = alpha;
		return this;
	}
	
	public float getAlpha () {
		return alpha;
	}
	
	public SolverConf setTau (int tau) {
		this.tau = tau;
		return this;
	}
	
	public int getTau () {
		return tau;
	}
	
	public float getBaseModelMomentum () {
		return baseModelMomentum;
	}
	
	public SolverConf setBaseModelMomentum (float baseModelMomentum) {
		this.baseModelMomentum = baseModelMomentum;
		return this;
	}
	
	public float getMinLearningRate () {
		return minLearningRate;
	}
	
	public SolverConf setMinLearningRate (float minLearningRate) {
		this.minLearningRate = minLearningRate;
		return this;
	}
	
	public float getMaxLearningRate () {
		return maxLearningRate;
	}
	
	public SolverConf setMaxLearningRate (float maxLearningRate) {
		this.maxLearningRate = maxLearningRate;
		return this;
	}
	
	public float getLearningRateIncrement () {
		return incLearningRate;
	}
	
	public SolverConf setLearningRateIncrement (float incLearningRate) {
		this.incLearningRate = incLearningRate;
		return this;
	}
	
	public boolean useSuperConvergence () {
		return superConvergence;
	}
	
	public SolverConf setSuperConvergence (boolean superConvergence) {
		this.superConvergence = superConvergence;
		return this;
	}
	
	public float getMinMomentum () {
		return minMomentum;
	}
	
	public SolverConf setMinMomentum (float minMomentum) {
		this.minMomentum = minMomentum;
		return this;
	}
	
	public float getMaxMomentum () {
		return maxMomentum;
	}
	
	public SolverConf setMaxMomentum (float maxMomentum) {
		this.maxMomentum = maxMomentum;
		return this;
	}
	
	public float getMomentumIncrement () {
		return incMomentum;
	}
	
	public SolverConf setMomentumIncrement (float incMomentum) {
		this.incMomentum = incMomentum;
		return this;
	}
	
	public void GPURegister () {
		
		TheGPU.getInstance().setEamsgdAlpha (alpha);
		TheGPU.getInstance().setEamsgdTau   (tau);
		
		TheGPU.getInstance().setMomentum    (momentum, momentumMethod.getId ());
		TheGPU.getInstance().setWeightDecay (weightDecay);
		
		/* Set learning rate */
		switch (learningRateDecayPolicy) {
		case FIXED: 
			TheGPU.getInstance().setLearningRateDecayPolicyFixed (baseLearningRate); 
			break;
		case INV:
			TheGPU.getInstance().setLearningRateDecayPolicyInv (baseLearningRate, gamma, power);
			break;
		case STEP:
			TheGPU.getInstance().setLearningRateDecayPolicyStep (baseLearningRate, gamma, getStepSize());
			break;
		case MULTISTEP:
			if (stepvalues == null)
				throw new  NullPointerException();
			TheGPU.getInstance().setLearningRateDecayPolicyMultiStep (baseLearningRate, gamma, getStepValues());
			break;
		case EXP:
			TheGPU.getInstance().setLearningRateDecayPolicyExp (baseLearningRate, gamma);
			break;
		case CLR:
			TheGPU.getInstance().setLearningRateDecayPolicyCircular (
					new float [] {
						minLearningRate, 
						maxLearningRate, 
						incLearningRate
					}, 
					useSuperConvergence() ? 1 : 0,
					new float [] {
						minMomentum, 
						maxMomentum, 
						incMomentum 
					},
					getStepSize());
		default:
			throw new IllegalStateException ("error: invalid learning rate decay policy type");
		}
		
		TheGPU.getInstance().setBaseModelMomentum (baseModelMomentum);
	}
	
	public static LinkedList<Option> getOptions () {
		
		LinkedList<Option> opts = new LinkedList<Option>();
		
		opts.add(new Option("--learning-rate-decay-policy").setType ( String.class));
		opts.add(new Option("--learning-rate"             ).setType (  Float.class));
		opts.add(new Option("--gamma"                     ).setType (  Float.class));
		opts.add(new Option("--power"                     ).setType (  Float.class));
		opts.add(new Option("--momentum"                  ).setType (  Float.class));
		opts.add(new Option("--momentum-method"           ).setType ( String.class));
		opts.add(new Option("--weight-decay"              ).setType (  Float.class));
		opts.add(new Option("--regularisation-type"       ).setType ( String.class));
		opts.add(new Option("--clip-gradient-threshold"   ).setType ( String.class));
		opts.add(new Option("--step-size"                 ).setType (Integer.class));
		opts.add(new Option("--step-values"               ).setType ( String.class));
		opts.add(new Option("--learning-rate-step-unit"   ).setType ( String.class));
		opts.add(new Option("--alpha"                     ).setType (  Float.class));
		opts.add(new Option("--tau"                       ).setType (Integer.class));
		opts.add(new Option("--base-model-momentum"       ).setType (  Float.class));
		opts.add(new Option("--min-learning-rate"         ).setType (  Float.class));
		opts.add(new Option("--max-learning-rate"         ).setType (  Float.class));
		opts.add(new Option("--learning-rate-increment"   ).setType (  Float.class));
		opts.add(new Option("--super-convergence"         ).setType (Boolean.class));
		opts.add(new Option("--min-momentum"              ).setType (  Float.class));
		opts.add(new Option("--max-momentum"              ).setType (  Float.class));
		opts.add(new Option("--momentum-increment"        ).setType (  Float.class));
		
		return opts;
	}
	
	public String toString () {
		
		StringBuilder s = new StringBuilder ("=== [Solver configuration dump] ===\n");
		
		s.append (String.format("Learning rate decay policy is '%s'\n", learningRateDecayPolicy.toString ()));
		s.append (String.format("Learning rate is %.5f\n", baseLearningRate));
		s.append (String.format("Gamma is %.5f\n", gamma));
		s.append (String.format("Power is %.5f\n", power));
		s.append (String.format("Momentum is %.5f\n", momentum));
		s.append (String.format("Momentum method is %s\n", momentumMethod));
		s.append (String.format("Weight decay is %.5f\n", weightDecay));
		s.append (String.format("Regularisation type is '%s'\n", regularizationType.toString()));
		s.append (String.format("Step size is %d\n", stepsize));
		s.append (String.format("Step values are %s\n", getStepValuesToString()));
		s.append (String.format("Alpha is %.5f\n", alpha));
		s.append (String.format("Tau is %d\n", tau));
		s.append (String.format("Base model momentum is %.5f\n", baseModelMomentum));
		s.append (String.format("Min learning rate is %.5f\n", minLearningRate));
		s.append (String.format("Max learning rate is %.5f\n", maxLearningRate));
		s.append (String.format("Learning rate increment is %.5f\n", incLearningRate));
		s.append (String.format("%s super-convergence\n", useSuperConvergence() ? "Use" : "Don't use"));
		s.append (String.format("Min momentum is %.5f\n", minMomentum));
		s.append (String.format("Max momentum is %.5f\n", maxMomentum));
		s.append (String.format("Momentum increment is %.5f\n", incMomentum));
		
		s.append("=== [End of solver configuration dump] ===\n");
		
		return s.toString();
	}
}
