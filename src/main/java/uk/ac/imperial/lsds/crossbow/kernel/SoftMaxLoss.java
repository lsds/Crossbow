package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class SoftMaxLoss extends Kernel {
	
	private final static Logger log = LogManager.getLogger (SoftMaxLoss.class);
	
	private LossConf conf;
	
	private LocalVariable theLabels;
	
	public SoftMaxLoss (LossConf conf) {
		this.conf = conf;
	}

	public SoftMaxLoss setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		/* The second input to this operator are the labels */
		
		Variable labels = new Variable("labels", new Shape (new int [] { input.getShape().get(0) }), false);
		theLabels = new LocalVariable (labels);
		
		log.debug(String.format("Input variable %s", labels.getName()));
		
		/* Configure the output shape: the output of a loss operator is a scalar */
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
		
		/* Are there any GPU-specific local variables? Yes, `losses` and `count` */
		memoryRequirements.setLocalGPUMemoryRequirements (labels.capacity());
		memoryRequirements.incLocalGPUMemoryRequirements (labels.capacity());
		
		return this;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 2 inputs, 2 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 2, 2, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		Variable [] labels = theLabels.getInitialValue();
		
		/* Set inputs */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Setup local variables */
		TheGPU.getInstance().setKernelLocalVariable (id, 0, "losses", labels[0].getShape().array(), labels[0].capacity(),  false);
		TheGPU.getInstance().setKernelLocalVariable (id, 1, "counts", labels[0].getShape().array(), labels[0].capacity(),  false);
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "ignorelabel", conf.getIgnoredLabelValue());
		
		return;
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		int ignoredLabelValue = conf.getIgnoredLabelValue ();
		boolean ignoreLabel = (ignoredLabelValue >= 0);
		
		/* Get input buffer */
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		
		/* Get labels */
		IDataBuffer labelsDataBuffer = batch.getInputBuffer(1);
		int labelsStartP = batch.getBufferStartPointer(1);
		
		Variable []  input  = theInput.get();
		Variable [] labels = theLabels.get();
		Variable [] output = theOutput.get();
		/*
		 * For every example in batch, the previous
		 * operator has computed the probability of
		 * each possible outcome.
		 */
		int numberOfLabels  = labels[0].getShape().countAllElements();
		int numberOfOutputs =  input[0].getShape().get(1);
		
		int count = 0;
		float loss = 0F;
		
		int labelValue;
		float probability;
		int offset;
		for (int i = 0; i < numberOfLabels; ++i) {
			
			labelValue = labelsDataBuffer.getInt((labelsStartP + (i * 4))); /* & 0xFF; if unsigned */
			
			if (ignoreLabel && (labelValue == ignoredLabelValue))
				continue;
			
			offset = (i * numberOfOutputs + labelValue) * input[0].getType().sizeOf();
			probability = inputDataBuffer.getFloat(offset);
			
			/* loss -= Math.log(Math.max(probability, Float.MIN_VALUE)); */
			loss -= Math.log(Math.max(probability, 1.175494e-38));
			
			++count;
		}
		
		/* Normalise loss value */
		loss /= getNormalisationValue(count, numberOfLabels);
		
		/* Get an output buffer */
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		
		outputDataBuffer.putFloat(0, loss);
		
		batch.setOutput(operator.getId(), outputDataBuffer);
		batch.setLoss (loss);
	}
	
	private float getNormalisationValue (int count, int total) {
		float v;
		switch (conf.getMode()) {
		case VALID:
			v = (float) count;
			break;
		case FULL:
		case BATCH:
			v = (float) total;
			break;
		case NONE:
			v = 1F;
			break;
		default:
			throw new IllegalStateException("error: unknown normalisation mode");
		}
		return Math.max(v, 1F);
	}
	
	public ModelAccess getModelAccessType () {
		return ModelAccess.NA;
	}
	
	public boolean isLossKernel () {
		return true;
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
