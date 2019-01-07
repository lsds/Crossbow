package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Dummy extends Kernel {
	
	private final static Logger log = LogManager.getLogger (Dummy.class);
	
	public Dummy () {
	}
	
	@Override
	public IKernel setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);

		log.debug(String.format("Input variable %s", input.getName()));

		/* Configure the output shape */

		outputShape = inputShape[0].copy();

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

	@Override
	public void GPURegister () {
	}

	@Override
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {	
	}
	
	@Override
	public ModelAccess getModelAccessType() {
		return ModelAccess.NA;
	}

	@Override
	public boolean isLossKernel () {
		return false;
	}

	@Override
	public boolean isAccuracyKernel () {
		return false;
	}

	@Override
	public boolean isDataTransformationKernel () {
		return false;
	}

	@Override
	public boolean allowsOutputOverwrite () {
		return false;
	}
	
	@Override
	public boolean allowsInputOverwrite () {
		return false;
	}
}
