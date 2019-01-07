package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class SoftMaxLossGradient extends Kernel {
	
	private final static Logger log = LogManager.getLogger (SoftMaxLossGradient.class);
	
	private LossConf conf;
	
	private LocalVariable theLabels;
	
	public SoftMaxLossGradient (LossConf conf) {
		this.conf = conf;
	}

	public SoftMaxLossGradient setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		/* Configure the output shape
		 * 
		 * The output of a gradient operator has the same 
		 * shape as the input of its forward peer.
		 * 
		 * But note that the peer could have more than one inputs.
		 */
		Operator peer = operator.getPeer();
		Shape [] p = peer.getInputShape();
		if (p.length > 1)
			throw new IllegalStateException(String.format("error: peer operator %s has more than one inputs", peer.getName()));
		
		outputShape = p[0].copy();
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		Variable labels = new Variable("labels", new Shape (new int [] { output.getShape().get(0) }), true);
		theLabels = new LocalVariable (labels);
		
		log.debug(String.format("Input variable %s", labels.getName()));
		
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
		
		/* Are there any GPU-specific local variables? Yes, `count` */
		memoryRequirements.setLocalGPUMemoryRequirements (labels.capacity());
		
		return this;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 2 inputs, 1 local variable, 1 output */
		TheGPU.getInstance().setKernel (id, name, 2, 1, 1, (isLossKernel() || isAccuracyKernel())); 
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		Variable [] labels = theLabels.getInitialValue();
		
		/* Set inputs */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Setup local variables */
		TheGPU.getInstance().setKernelLocalVariable (id, 0, "count", labels[0].getShape().array(), labels[0].capacity(), false);
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "ignorelabel", conf.getIgnoredLabelValue());
		
		return;
	}
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable [] output = theOutput.get();
		Variable [] labels = theLabels.get();
		
		/*
		 * Get input of peer operator. Input contains the class 
		 * probabilities computed by Softmax
		 */
		IDataBuffer peerInputDataBuffer = getPeerInput (batch, api);
		int peerInputStartP = getStartPointer ();
		int peerInputEndP = getEndPointer ();
		
		/* Get labels */
		IDataBuffer labelsDataBuffer = batch.getInputBuffer(1);
		int labelsStartP = batch.getBufferStartPointer(1);
		
		/* Initialise output buffer `C` by copying buffer `A` */
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		outputDataBuffer.put (peerInputDataBuffer, peerInputStartP, peerInputEndP - peerInputStartP, true);
		
		int ignoredLabelValue = conf.getIgnoredLabelValue();
		boolean ignoreLabel = (ignoredLabelValue >= 0);
		
		int numberOfClasses = output[0].getShape().get(1); /* `output` should be a 2-D variable; */
		int numberOfLabels  = labels[0].getShape().get(0); /* `labels` is a 1-D variable */
		
		int count = 0;
		
		int labelValue;
		int offset;
		
		for (int i = 0; i < numberOfLabels; ++i) {
			/* Get label */
			labelValue = labelsDataBuffer.getInt((labelsStartP + (i * 4))); // & 0xFF;
			/* Compute class value offset */
			offset = (i * numberOfClasses + labelValue) * output[0].getType().sizeOf();
			/* Check if user ignores this label */
			if (ignoreLabel && (labelValue == ignoredLabelValue)) {
				outputDataBuffer.putFloat(offset, 0F);
			} else {
				outputDataBuffer.putFloat(offset, (outputDataBuffer.getFloat(offset) - 1));
				++count;
			}
		}
		
		float _loss_weight = 1F / getNormalisationValue (count, numberOfLabels);
		
		/* Normalise output */
		IDataBufferIterator iterator = outputDataBuffer.getIterator();
		while (iterator.hasNext()) {
			offset = iterator.next();
			outputDataBuffer.putFloat(offset, (_loss_weight * outputDataBuffer.getFloat(offset)));
		}
		
		/* Store output */
		batch.setOutput(operator.getId(), outputDataBuffer);
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
