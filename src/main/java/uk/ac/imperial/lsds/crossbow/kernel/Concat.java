package uk.ac.imperial.lsds.crossbow.kernel;

import java.util.Arrays;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConcatConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Concat extends Kernel {
	
	private final static Logger log = LogManager.getLogger (Concat.class);
	
	private ConcatConf conf;
	
	public Concat (ConcatConf conf) {
		this.conf = conf;
	}
	
	public IKernel setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length < 2)
			throw new IllegalStateException (String.format("error: invalid number of inputs in operator %s", operator.getName()));
				
		Variable [] input = new Variable [inputShape.length];
		for (int i = 0; i < inputShape.length; ++i)
			input [i] = new Variable (String.format("input-%d", (i + 1)), inputShape[i], true);
		
		theInput = new LocalVariable (input);
		
		for (int i = 0; i < inputShape.length; ++i)
			log.debug(String.format("Input variable %s", input[i].getName()));
		
		int axis = conf.getAxis();
		int channels = inputShape[0].get(axis);
		int elements = inputShape[0].countAllElements();
		
		for (int i = 1; i < inputShape.length; ++i) {
			
			/* Check that all inputs have the same number of dimensions */
			if (inputShape[0].dimensions() != inputShape[i].dimensions())
				throw new IllegalStateException 
					(String.format("error: invalid input dimensions in operator %s", operator.getName()));
			
			/* Check that all inputs have the same shape, other than the concatenation axis */
			for (int j = 0; j < inputShape[0].dimensions(); ++j) {
				if (j == axis)
					continue;
				if (inputShape[0].get(j) != inputShape[i].get(j))
					throw new IllegalStateException 
					(String.format("error: invalid input shape in operator %s (%s vs %s)", operator.getName(), inputShape[0].toString(), inputShape[i].toString()));
			}
			
			channels += inputShape[i].get(axis);
			elements += inputShape[i].countAllElements();
		}
		
		/* Configure the output shape */
		
		int [] s = Arrays.copyOf(inputShape[0].array(), inputShape[0].dimensions());
		s[axis] = channels;
		
		outputShape = new Shape (s);
		
		if (outputShape.countAllElements() != elements)
			throw new IllegalStateException 
				(String.format("error: invalid output shape in operator %s", operator.getName()));
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));
		
		/* Set memory requirements */
		
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
		
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis", conf.getAxis());
		
		return;
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
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
