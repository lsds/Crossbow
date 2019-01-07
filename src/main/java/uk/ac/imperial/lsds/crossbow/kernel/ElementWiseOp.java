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

public class ElementWiseOp extends Kernel {
	
	private final static Logger log = LogManager.getLogger (ElementWiseOp.class);
	
	private ElementWiseOpConf conf;
	
	public ElementWiseOp (ElementWiseOpConf conf) {
		
		this.conf = conf;
	}
	
	@Override
	public IKernel setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		int numberOfInputs = inputShape.length;
		
		/* Check that the input shapes are all the same */
		if (numberOfInputs > 1) {
			Shape shape = inputShape [0];
			log.debug(String.format("Input shape [0] is %s", shape));
			for (int i = 1; i < numberOfInputs; ++i) {
				log.debug(String.format("Input shape [%d] is %s", i, inputShape[i]));
				if (shape.countAllElements() != inputShape[i].countAllElements())
					throw new IllegalStateException (String.format("error: input shape mismatch in operator %s", operator.getName()));
			}
		}
		
		Variable [] input = new Variable [numberOfInputs];
		for (int i = 0; i < numberOfInputs; ++i)
			input [i] = new Variable (String.format("input-%d", (i + 1)), inputShape[i], true);
		
		theInput = new LocalVariable (input);
		
		for (int i = 0; i < numberOfInputs; ++i)
			log.debug(String.format("Input variable %s", input[i].getName()));
		
		/* Resize coefficients */
		conf.resizeCoefficients(numberOfInputs);
		
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
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		/* Variable number of inputs, 0 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, input.length, 0, 1, (isLossKernel() || isAccuracyKernel()));
		
		/* Set input */
		for (int i = 0; i < input.length; ++i)
			TheGPU.getInstance().setKernelInput (id, i, input[i].getShape().array(), input[i].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id, output[0].getShape().array(), output[0].capacity());
		
		/* Set GPU kernel-specific parameters */
		
		/* TODO: If we are to support operations other than sum, we should also specify a type */
		
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsFloatArray (id, 0, "coefficients", conf.getCoefficients());
		
		return;
	}
	
	@Override
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (conf.getType() != ElementWiseOpType.SUM)
			throw new IllegalStateException (String.format("error: element-wise operation %s is not supported yet", conf.getType().toString()));
		
		Variable []  input =  theInput.get();
		Variable [] output = theOutput.get();
		
		if (previous.length != input.length)
			throw new IllegalStateException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		/* Fill output with 0s */
		outputDataBuffer.bzero();
		
		float [] coefficients = conf.getCoefficients();
		int elements = input[0].getShape().countAllElements();
		
		IDataBuffer inputDatabuffer;
		int inputStartP, inputEndP;
		
		for (int i = 0; i < input.length; ++i) {
			
			/* Get i-th input buffer */
			inputDatabuffer = getOperatorOutput(previous[i], batch, api);
			inputStartP = getStartPointer ();
			inputEndP = getEndPointer ();
			
			BLAS.getInstance().saxpby(elements, coefficients[i], inputDatabuffer, inputStartP, inputEndP, 1, 1F, outputDataBuffer, 1);
		}
		
		/* Store output in batch for downstream operators */
		batch.setOutput(operator.getId(), outputDataBuffer);
		
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
