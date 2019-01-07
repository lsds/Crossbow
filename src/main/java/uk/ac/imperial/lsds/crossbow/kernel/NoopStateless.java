package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.NoopConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class NoopStateless extends Kernel {

	private final static Logger log = LogManager.getLogger (NoopStateless.class);

	@SuppressWarnings("unused")
	private NoopConf conf;
	
	public NoopStateless (NoopConf conf) {
		this.conf = conf;
	}
	
	public NoopStateless setup (Shape [] inputShape, Model model) {

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
		
		/* Set memory requirements */
		
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		return this;
	}

	public void GPURegister () {

		int id = operator.getId();
		String name = this.getClass().getSimpleName();

		/* 1 input, 0 local variables, 1 output, pull = false */
		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));

		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();

		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());

		return;
	}
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		Variable []  input =  theInput.get();
		Variable [] output = theOutput.get();
		
		log.debug(String.format("Input  variable shape is %s",  input[0].getShape()));
		log.debug(String.format("Output variable shape is %s", output[0].getShape()));
		
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		int inputEndP = getEndPointer ();
		
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer); 
		
		/* Copy input to output */
		log.debug(String.format("In operator %d, put %d bytes, starting from %d", operator.getId(), inputEndP - inputStartP, inputStartP));
		// if (operator.getId() == 0)
		outputDataBuffer.put(inputDataBuffer, inputStartP, inputEndP - inputStartP, true);
		
		batch.setOutput (operator.getId(), outputDataBuffer);
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
