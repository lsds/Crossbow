package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.AccuracyConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Accuracy extends Kernel {

	private final static Logger log = LogManager.getLogger (Accuracy.class);

	private AccuracyConf conf;

	private LocalVariable theLabels;

	public Accuracy (AccuracyConf conf) {
		this.conf = conf;
	}

	public Accuracy setup (Shape [] inputShape, Model model) {

		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable input = new Variable ("input", inputShape[0], true);
		this.theInput = new LocalVariable (input);

		log.debug(String.format("Input variable %s", input.getName()));

		/* The second input to this operator are the labels */
		Variable labels = new Variable("labels", new Shape (new int [] { input.getShape().get(0) }), true);
		theLabels = new LocalVariable (labels);

		log.debug(String.format("Input variable %s", labels.getName()));
		
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
		Variable [] labels = theLabels.getInitialValue();
		
		/* Set inputs */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
		
		/* Set output. The output is a single value */
		TheGPU.getInstance().setKernelOutput (id, new int [] { 1 }, 4);
		
		/* Set local variable */
		TheGPU.getInstance().setKernelLocalVariable (id, 0, "count", labels[0].getShape().array(), labels[0].capacity(),  false);
		
		return;
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {

		log.debug(String.format("Compute kernel for operator %s", operator.getName()));

		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		int ignoredLabelValue = conf.getIgnoredLabelValue();
		boolean ignoreLabel = (ignoredLabelValue >= 0);

		int axis = conf.getAxis();

		/* Get input */
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();

		/* Get labels */
		IDataBuffer labelsDataBuffer = batch.getInputBuffer(1);
		int labelsStartP = batch.getBufferStartPointer(1);

		Variable []  input = theInput.get();
		Variable [] labels = theLabels.get();
		
		/*
		 * For every example in batch, the previous
		 * operator has computed the probability of
		 * each possible outcome.
		 * 
		 * Analyse and check input dimensions.
		 */
		int numberOfLabels   = labels [0].getShape().countAllElements();
		int numberOfExamples = input  [0].getShape().countElements(0, axis);
		int numberOfClasses  = input  [0].getShape().get(axis);
		
		int outer = input[0].getShape().countAllElements() / numberOfExamples;
		int inner = input[0].getShape().countElements(axis + 1);
		
		if ((numberOfLabels != numberOfExamples) || (inner != 1) || (outer != numberOfClasses)) {

			System.err.println(String.format("error: unexpected dimensions in variable %s", input[0].getName()));
			System.exit(1);
		}
		
		int accuracyCount = 0;

		for (int i = 0; i < numberOfExamples; ++i) {

			int labelValue = labelsDataBuffer.get(labelsStartP + i) & 0xFF;

			if (ignoreLabel && (labelValue == ignoredLabelValue))
				continue;

			/* Find maximum by iterating over class values */
			float maxValue = Float.MIN_VALUE;
			float maxLabel = -1;

			for (int j = 0; j < numberOfClasses; ++j) {

				int pos = inputStartP + (i * numberOfClasses + j) * input[0].getType().sizeOf();

				float classValue = inputDataBuffer.getFloat(pos);

				if (maxValue < classValue) { 
					maxValue = classValue;
					maxLabel = j;
				}
			}
			
			if (maxLabel < 0) {
				System.err.println(String.format("error: invalid classification label"));
				System.exit(1);
			}

			if (maxLabel == labelValue)
				accuracyCount ++;
		}
		
		float accuracy = (float) accuracyCount / (float) numberOfExamples;
		log.info(String.format("Accuracy %10.5f", accuracy));

		batch.setAccuracy (accuracy);
	}
	
	public ModelAccess getModelAccessType () {
		return ModelAccess.NA;
	}
	
	public boolean isLossKernel () {
		return false;
	}
	
	public boolean isAccuracyKernel () {
		return true;
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
