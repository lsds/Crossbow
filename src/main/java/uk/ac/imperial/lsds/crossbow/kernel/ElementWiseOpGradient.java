package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ElementWiseOpConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ElementWiseOpType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class ElementWiseOpGradient extends Kernel {
	
	private final static Logger log = LogManager.getLogger (ElementWiseOpGradient.class);
	
	private ElementWiseOpConf conf;
	
	public ElementWiseOpGradient (ElementWiseOpConf conf) {
		
		this.conf = conf;
	}
	
	@Override
	public IKernel setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		/* Configure the input shape */
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		/* Resize coefficients */
		conf.resizeCoefficients(inputShape.length);
		
		/* Configure the output shape */

		outputShape = inputShape[0].copy();

		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);

		log.debug(String.format("Output variable %s", output.getName()));

		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		/* memoryRequirements.setOutputMemoryRequirements(output.capacity() * operator.getPeer().numberOfInputs ()); */
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
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		/* 1 input, 0 local variables, variable outputs, pull = false */
		/* TheGPU.getInstance().setKernel (id, name, 1, 0, operator.getPeer().numberOfInputs(), (isLossKernel() || isAccuracyKernel())); */
		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
		
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsFloatArray (id, 0, "coefficients", conf.getCoefficients());
		
		return;
	}
	
	@Override
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (conf.getType() != ElementWiseOpType.SUM)
			throw new IllegalStateException (String.format("error: element-wise operation %s is not supported yet", conf.getType().toString()));
		
		/* Variable [] input =  theInput.get(); */
		Variable [] output = theOutput.get();
		
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		int inputEndP = getEndPointer ();
		
		float [] coefficients = conf.getCoefficients ();
		int elements = output[0].getShape().countAllElements();
		
		int numberOfOutputs = 1; /* operator.getPeer().numberOfInputs (); */
		
		for (int i = 0; i < numberOfOutputs; ++i) {
			
			/* TODO Handle getCurrentOuput when we have multiple outputs */
			IDataBuffer outputbuffer = getCurrentOutput (batch, api);
			output[0].wrap(outputbuffer);
			/* Instead of outputbuffer.bzero(), set beta = 0. */
			BLAS.getInstance().saxpby(elements, coefficients[i], inputDataBuffer, inputStartP, inputEndP, 1, 0F, outputbuffer, 1);
			
			/* Store output in batch for downstream operators */
			batch.setOutput(operator.getId(), outputbuffer);
		}
		
		return;
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
