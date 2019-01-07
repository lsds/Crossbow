package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.DropoutConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.CudnnKernelType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Dropout extends Kernel {
	
	private final static Logger log = LogManager.getLogger (Dropout.class);
	
	DropoutConf conf;

	public Dropout (DropoutConf conf) {
		this.conf = conf;
	}
	
	public Dropout setup (Shape [] inputShape, Model model) {
		
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
		
		/* Are there any GPU-specific local variables? Yes, but they cannot be defined at the moment */
		memoryRequirements.setLocalGPUMemoryRequirements (0);
		
		return this;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));

		int id = operator.getId();
		String name = this.getClass().getSimpleName();

		/* 1 input, 0 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 1, 1, (isLossKernel() || isAccuracyKernel()));

		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Set cuDNN kernel */
		TheGPU.getInstance().cudnnSetKernelType (id, CudnnKernelType.DROPOUT.getId());
		
		int [] dimensions = new int [4];
		
		input[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelInputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		output[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		int reserveSpaceSizeInBytes = TheGPU.getInstance().cudnnGetDropoutReserveSpaceSize (id);
		
		log.debug (String.format("Reserve space size is %d bytes", reserveSpaceSizeInBytes));
		
		TheGPU.getInstance().cudnnSetDropoutDescriptor (id, conf.getRatio(), SystemConf.getInstance().getRandomSeed());
		
		/* Set local variables */
		if (reserveSpaceSizeInBytes > 0) {
			TheGPU.getInstance().setKernelLocalVariable (id, 0, "reserveSpace", 
					new int [] { (reserveSpaceSizeInBytes / 4) }, reserveSpaceSizeInBytes, false);
			
			memoryRequirements.setLocalGPUMemoryRequirements (reserveSpaceSizeInBytes);
		}
	}
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
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
