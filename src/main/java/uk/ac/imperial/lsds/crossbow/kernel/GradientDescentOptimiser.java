package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.ModelGradient;
import uk.ac.imperial.lsds.crossbow.model.ModelIterator;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.model.VariableGradient;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;
import uk.ac.imperial.lsds.crossbow.types.Regularisation;

public class GradientDescentOptimiser extends Kernel {
	
	private final static Logger log = LogManager.getLogger (GradientDescentOptimiser.class);

	private SolverConf conf;
	
	public GradientDescentOptimiser (SolverConf conf) {
		this.conf = conf;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 1 inputs, 0 local variables, 1 output, pull = false */
		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
		
		TheGPU.getInstance().setKernelInput  (id, 0, new int [] {1}, 4);
		TheGPU.getInstance().setKernelOutput (id,    new int [] {1}, 4);
		
		return;
	}
	
	public GradientDescentOptimiser setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);

		log.debug(String.format("Input variable %s", input.getName()));

		/* Configure the output shape */

		outputShape = new Shape (new int [] { 1 });
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);

		log.debug(String.format("Output variable %s", output.getName()));

		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? No */
		memoryRequirements.setLocalCPUMemoryRequirements (0);
		
		/* Are there any GPU-specific local variables? No */
		memoryRequirements.setLocalGPUMemoryRequirements (0);
		
		return this;
	}
	
	public void compute (Operator[] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		ModelGradient gradient = batch.getModelGradient();
		
		float rate = getLearningRate(batch.getId());
		log.debug(String.format("Learning rate for batch %d is %5.5f", batch.getId(), rate));
		
//		Model theModel = model.getBaseModel();
		
		switch(ModelConf.getInstance().getUpdateModel()) {
		case DEFAULT:
		
			model.writeLock();
			
			clipGradient        (       gradient);
			applyWeightDecay 	(model, gradient); /* This method requires access to model variables */
			applyLearningRate   (rate,  gradient);
			applyMomentum       (model, gradient); /* This requires access to the last gradient that was applied to the model */
			
			/* Apply gradient to local model */
			model.apply(gradient);
			
			/* Unlock the model */
			model.writeUnlock();
			
//			/* Apply gradient to parameter server model (base model) */
//			 theModel.writeLock();
//			 theModel.apply(gradient);
//			 theModel.writeUnlock();
			
			break;
		case EAMSGD:
			/* Update replica using EAMSGD model */
			break;
		case SYNCHRONOUSEAMSGD:
			/* Update replica using synchronous EAMSGD model */
			
			model.writeLock();
			
			clipGradient        (       gradient);
			applyWeightDecay 	(model, gradient); /* This method requires access to model variables */
			applyLearningRate   (rate,  gradient);
			applyMomentum       (model, gradient); /* This requires access to the last gradient that was applied to the model */
			
			/* Apply gradient to local model */
			model.apply(gradient);
			
			/* Unlock the model */
			model.writeUnlock();
			break;
		case WORKER:
			/* Update replica using worker model */
			break;
		default:
			throw new IllegalArgumentException("error: invalid model update type");
		}
			
		return;
	}
	
	private float getLearningRate (int batchid) {
		
		float rate;
		
		float base  = conf.getBaseLearningRate();
		float gamma = conf.getGamma();
		float power = conf.getPower();
		float stepsize = conf.getStepSize();
		int [] stepvalue = conf.getStepValues();
		int currentstep = conf.getCurrentStep();
		
		LearningRateDecayPolicy policy = conf.getLearningRateDecayPolicy();
		switch (policy) {
		
		case FIXED:
			rate = base;
			break;
			
		case INV:
			rate = base * (float) Math.pow (1F + gamma * ((float) (batchid + 1)), -power);
			break;
		
		case STEP:
			if (stepsize == 0)
				throw new IllegalArgumentException("error: step size is not defined");

			rate = base * (float) Math.pow (gamma, Math.floor((batchid + 1)/ stepsize));
			break;
			
		case MULTISTEP:
			if (stepvalue == null)
				throw new IllegalArgumentException("error: step value array is not predefined");
			
			if ((currentstep < stepvalue.length) && ((batchid + 1) >= stepvalue[currentstep])){
				currentstep++;
				conf.setCurrentStep(currentstep);
			}
			rate = base * (float) Math.pow (gamma, currentstep);
			
			break;
			
		case EXP:
			rate = base * (float) Math.pow (gamma, (batchid + 1));
			break;
		
		default:
			throw new IllegalArgumentException("error: invalid learning rate decay policy");
		}
		
		return rate;
	}
	
	private void clipGradient (ModelGradient gradient) {
		
		float threshold = conf.getClipGradientThreshold();
		
		if (threshold < 0) {
			return;
		}
		
		/* Compute L2 norm */
		
		float L2, sumsquared = 0F;
		
		ModelIterator<VariableGradient> i = gradient.iterator();
		IDataBuffer buffer;
		
		IDataBufferIterator j;
		int offset;
		
		while (i.hasNext()) {
			
			buffer = i.next().getDataBuffer();
			j = buffer.getIterator();
			
			while (j.hasNext()) {
				
				offset = j.next();
				sumsquared += (float) Math.pow (buffer.getFloat(offset), 2);
			}
		}
		
		L2 = (float) Math.sqrt(sumsquared);
		
		if (L2 > threshold) { // Scale gradient
			
			float factor = threshold / L2;
			i.reset();
			
			while (i.hasNext()) {
				
				buffer = i.next().getDataBuffer();
				j = buffer.getIterator();
				
				while (j.hasNext()) {
					
					offset = j.next();
					buffer.putFloat (offset, factor * buffer.getFloat(offset));
				}
			}
		}
		
		return;
	}
	
	private void applyWeightDecay (Model model, ModelGradient gradient) {
		
		float decay = conf.getWeightDecay();
		
		if (decay <= 0F)
			return;
		
		ModelIterator<Variable> m = model.iterator();
		ModelIterator<VariableGradient> g = gradient.iterator();
		
		IDataBuffer X, Y;
		int N;
		
		Regularisation t = conf.getRegularisationType();
		switch (t) {
		
		case L2:
			while (m.hasNext() && g.hasNext()) {
				X = m.next().getDataBuffer();
				Y = g.next().getDataBuffer();
				
				N = X.limit() / DataType.FLOAT.sizeOf();
				
				BLAS.getInstance().saxpby(N, decay, X, 0, X.limit(), /* incX */ 1, 1F, Y, /* incY */ 1);
			}
			break;
			
		case L1:
			throw new IllegalArgumentException("error: L1 gradient normalisation not yet implemented");
			
		default:
			throw new IllegalArgumentException("error: invalid gradient regularisation type");
		}
	}
	
	private void applyLearningRate (float rate, ModelGradient gradient) {
		
		/* Apply learning rate to gradient */
		
		ModelIterator<VariableGradient> g = gradient.iterator();
		VariableGradient var;
		IDataBuffer buffer;
		IDataBufferIterator b;
		int offset;
		
		while (g.hasNext()) {
			
			var = g.next();
			buffer = var.getDataBuffer();
			b = buffer.getIterator();
			
			while (b.hasNext()) {
				
				offset = b.next();
				buffer.putFloat(offset, var.getLearningRateMultiplier() * rate * buffer.getFloat(offset));
			}
		}
	}
	
	private void applyMomentum (Model model, ModelGradient gradient) {
		
		/* Apply momentum */
		
		float momentum = conf.getMomentum();
		
		if (momentum == 0F) {
			return;
		}
		
		// model.setLastGradientRequirement(true);

		ModelGradient last = model.getLastGradient();
		
		if (last == null) {
			return;
		}
		
		ModelIterator<VariableGradient> g = gradient.iterator();
		ModelIterator<VariableGradient> l =     last.iterator();
		
		IDataBuffer X, Y;
		int N;
		
		while (l.hasNext() && g.hasNext()) {
			
			X = l.next().getDataBuffer();
			Y = g.next().getDataBuffer();
			
			N = X.limit() / DataType.FLOAT.sizeOf();
			
			BLAS.getInstance().saxpby(N, momentum, X, 0, X.limit(), /* incX */ 1, 1F, Y, /* incY */ 1);
		}	
	}

	public ModelAccess getModelAccessType () {
		return ModelAccess.RW;
	}
	
	public boolean isLossKernel () {
		return false;
	}
	
	public boolean isAccuracyKernel () {
		return false;
	}
	
	public boolean isDataTransformationKernel () {
		return false;
	}
	
	public boolean allowsOutputOverwrite () {
		return false;
	}
	
	public boolean allowsInputOverwrite () {
		return false;
	}
}
