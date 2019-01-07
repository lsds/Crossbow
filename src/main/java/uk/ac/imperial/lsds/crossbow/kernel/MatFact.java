package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.MatFactConf;
import uk.ac.imperial.lsds.crossbow.model.*;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class MatFact extends Kernel {

	private final static Logger log = LogManager.getLogger (MatFact.class);

	private MatFactConf conf;

	private LocalVariable theLabels;

	public MatFact (MatFactConf conf) {
		this.conf = conf;
	}

	public IKernel setup (Shape [] inputShape, Model model) {

		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable(input);

		log.debug(String.format("Input variable %s", input.getName()));
		
		/* The second input to this operator are the labels */

		Variable labels = new Variable("labels", new Shape (new int [] { input.getShape().get(0) }), false);
		theLabels = new LocalVariable (labels);

		log.debug(String.format("Input variable %s", labels.getName()));
		
		/* Configure the output shape: it is just a scalar (loss) */
		outputShape = new Shape (new int [] { 1 });
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));

		/* Register model variables */
		
		int numberOfLatentVariables = conf.numberOfLatentVariables ();
		int numberOfUsers = conf.numberOfRows ();
		int numberOfItems = conf.numberOfColumns ();
		
		Variable users = new Variable ("user", new Shape (new int [] { numberOfUsers, numberOfLatentVariables } ), false);
		users.initialise (conf.getModelVariableInitialiser().setValue(1f));
		model.register (operator.getId(), users);
		
		Variable items = new Variable ("item", new Shape (new int [] { numberOfItems, numberOfLatentVariables } ), false);
		items.initialise (conf.getModelVariableInitialiser().setValue(1f));
		model.register (operator.getId(), items);
		
		/* Set memory requirements */

		memoryRequirements.setOutputMemoryRequirements(output.capacity());

		memoryRequirements.setModelMemoryRequirements(users.capacity());
		memoryRequirements.incModelMemoryRequirements(items.capacity());
		
		memoryRequirements.setLocalGPUMemoryRequirements(labels.capacity());

		return this;
	}

	public void GPURegister () {

		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));

		int id = operator.getId();
		String name = this.getClass().getSimpleName();

		TheGPU.getInstance().setKernel (id, name, 2, 1, 1, (isLossKernel() || isAccuracyKernel()));

		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		Variable [] labels = theLabels.getInitialValue();

		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());

		/* Configure local variable */

		TheGPU.getInstance().setKernelLocalVariable (id, 0, "loss", labels[0].getShape().array(), labels[0].capacity(), false);

		/* Configure GPU kernel-specific parameters */

		TheGPU.getInstance().setKernelConfigurationParameters (id, 5);

		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 0, "latents", conf.numberOfLatentVariables());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 1,    "rows", conf.numberOfRows());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 2, "columns", conf.numberOfColumns());
		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 3,  "lambda", conf.getLambda());
		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 4,    "rate", conf.getLearningRateEta0());
		
		return;
	}
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {

		log.debug(String.format("Compute kernel for operator %s", operator.getName()));

		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		/* Get two variables from model and their gradients */

		Variable users = model.getVariable(operator.getId(), 1);
		Variable items = model.getVariable(operator.getId(), 2);
		
		IDataBuffer usersDataBuffer = users.getDataBuffer();
		IDataBuffer itemsDataBuffer = items.getDataBuffer();
		
		ModelGradient gradient = batch.getModelGradient(model);
		
		VariableGradient usersGradient = gradient.getVariableGradient (operator.getId(), 1);
		VariableGradient itemsGradient = gradient.getVariableGradient (operator.getId(), 2);
		
		IDataBuffer usersGradientDataBuffer = usersGradient.getDataBuffer();
		IDataBuffer itemsGradientDataBuffer = itemsGradient.getDataBuffer();
		
		/* Reset gradients */
		usersGradientDataBuffer.bzero();
		itemsGradientDataBuffer.bzero();
		
		/* Configure inputs (examples and labels) */
		Variable [] input = theInput.get();

		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		int inputEndP = getEndPointer ();
		
		IDataBuffer labelsDataBuffer = batch.getInputBuffer (1);
		int labelsStartP = batch.getBufferStartPointer (1);
		int labelsEndP = batch.getBufferEndPointer (1);
		
		log.debug(String.format("Task %d: examples [%d, %d), labels [%d, %d)", batch.getId(), inputStartP, inputEndP, labelsStartP, labelsEndP));
		
		int examples = input[0].getShape().numberOfExamples ();
		
		int K = conf.numberOfLatentVariables ();
		
		float rate = conf.getLearningRateEta0 ();
		float lambda = conf.getLambda ();
		
		/*
		 * Users and items are 2-D variables. The number of columns is 
		 * stored in the 2nd dimension.
		 */
		int usersRowSize = users.getShape().get(1) * users.getType().sizeOf();
		int itemsRowSize = items.getShape().get(1) * items.getType().sizeOf();
		
		float batchLoss = 0;
		
		/* Write-lock the model */
		model.writeLock();
		
		for (int idx = 0; idx < examples; ++idx) {
			
			int offset = inputStartP + (idx * 8); /* Each example is 8 bytes long */
			
			int userId = inputDataBuffer.getInt (offset);
			int itemId = inputDataBuffer.getInt (offset + 4);
			
			float rating = labelsDataBuffer.getFloat(labelsStartP + (idx * 4)); /* Each label is 4 bytes long */
			
			/* log.debug(String.format("x = %5d y = %5d v = %5.5f", userId, itemId, rating)); */
			
			int rowOffset = userId * usersRowSize;
			int colOffset = itemId * itemsRowSize;
			
			/* Compute dot product */
			float product = 0;
			for (int k = 0; k < K; ++k) {
				product += (usersDataBuffer.getFloat(rowOffset + k * 4) * itemsDataBuffer.getFloat(colOffset + k * 4));
			}
			
			/* log.info(String.format("x = %5d y = %5d v = %5.5f estimated %5.5f", rowId, colId, rating, dotproduct)); */

			float error = rating - product;
			float loss = (error * error);
			batchLoss += loss;
			
			/* Compute gradient */
			
			for (int k = 0; k < K; k++) {
				
				float userModelValue = usersDataBuffer.getFloat (rowOffset + k * 4);
				float itemModelValue = itemsDataBuffer.getFloat (colOffset + k * 4);
				
				float userGradientValue = usersGradientDataBuffer.getFloat (rowOffset + k * 4);
				float itemGradientValue = itemsGradientDataBuffer.getFloat (colOffset + k * 4);
				
				/* Write to gradient buffer... */
				usersGradientDataBuffer.putFloat (rowOffset + k * 4, userGradientValue - rate * (2 * error * itemModelValue - 2 * lambda * userModelValue));
				itemsGradientDataBuffer.putFloat (colOffset + k * 4, itemGradientValue - rate * (2 * error * userModelValue - 2 * lambda * itemModelValue));
			}
		}
		
		model.apply(gradient);
		
		/* Unlock the model */
		model.writeUnlock();
		
		/* 
		 * TODO Compute batch loss as the average? Jbosen does not average, 
		 * so we don't either. If yes, then
		 * 
		 * batchloss /= (float) examples;
		 */
		batch.setLoss(batchLoss);
		log.debug(String.format("Batch %d loss %.5f", batch.getId(), batchLoss));
	}

	public ModelAccess getModelAccessType() {
		return ModelAccess.RW;
	}

	public boolean isLossKernel() {
		return true;
	}

	public boolean isAccuracyKernel() {
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
