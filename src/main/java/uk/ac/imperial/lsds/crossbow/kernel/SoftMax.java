package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.CudnnKernelType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class SoftMax extends Kernel {

	private final static Logger log = LogManager.getLogger (SoftMax.class);

	private SoftMaxConf conf;

	private LocalVariable example;

	public SoftMax (SoftMaxConf conf) {
		this.conf = conf;
	}
	
	public SoftMax setup (Shape [] inputShape, Model model) {

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

		/* Local variable(s) */
		
		Variable var = new Variable ("example", new Shape (new int [] { input.getShape().get(conf.getAxis()) }), false);
		example = new LocalVariable (var);

		log.debug(String.format("Local variable %s", var.getName()));
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? Yes, `cpu-example` */
		memoryRequirements.setLocalCPUMemoryRequirements (var.capacity());
		
		/* Are there any GPU-specific local variables? No */
		memoryRequirements.setLocalGPUMemoryRequirements (0);
		
		return this;
	}

	public void GPURegister () {

		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));

		int id = operator.getId();
		String name = this.getClass().getSimpleName();

		/* 1 input, 0 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));

		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();

		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis", conf.getAxis());
		
		/* Set cuDNN kernel */
		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.SOFTMAX.getId());

		int [] dimensions = new int [4];

		input[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);

		output[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		return;
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {

		log.debug(String.format("Compute kernel for operator %s", operator.getName()));

		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable []  input =  theInput.get();
		Variable [] output = theOutput.get();
		Variable []  local =   example.get();
		
		int axis = conf.getAxis();
		
		int examples = input[0].getShape().countElements(0, axis);
		int classes  = input[0].getShape().get(axis);
		int outputs  = input[0].getShape().countAllElements() / examples;
		int inner    = input[0].getShape().countElements(axis + 1);
		
		if (inner != 1 || classes != outputs) {
			System.err.println(String.format("error: unexpected number of dimensions in variable %s", input[0].getName()));
			System.exit(1);
		}

		/* Get input buffer */
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		int inputEndP = getEndPointer ();
		
		if (input[0].capacity() != (inputEndP - inputStartP)) {
			System.err.println(String.format("error: invalid buffer size for variable %s", input[0].getName()));
			System.exit(1);
		}
		
		/* Get an output buffer */
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		
		/* Reset position */
		outputDataBuffer.reset();
		
		log.debug(String.format("Output data buffer length is %d", outputDataBuffer.limit()));
		
		/* Get local variable buffer */
		IDataBuffer exampleDataBuffer = local[0].getDataBuffer();

		
		int length = exampleDataBuffer.limit();
		int offset = inputStartP;
		int pos;
		float max, classValue;
		IDataBufferIterator iterator;
		
		/* Iterate over each example in the batch */
		for (int i = 0; i < examples; ++i) {

			/* Clear previous example from the buffer; and put a new one. */
			// exampleDataBuffer.reset();
			exampleDataBuffer.put(inputDataBuffer, offset, length, true);

			/* Find maximum by iterating over class values */
			max = Float.MIN_VALUE;

			for (int j = 0; j < classes; ++j) {

				pos = (i * classes + j) * input[0].getType().sizeOf();

				classValue = inputDataBuffer.getFloat(pos);

				if (max < classValue) 
					max = classValue;
			}

			/* 
			 * Subtract max from this example's (initialised) output and exponentiate.
			 * Meanwhile, maintain a sum (to be used for normalisation).
			 */
			float v, normalisationValue = 0F;

			iterator = exampleDataBuffer.getIterator();
			while (iterator.hasNext()) {

				pos = iterator.next();

				v = (float) Math.exp(exampleDataBuffer.getFloat(pos) - max);

				exampleDataBuffer.putFloat(pos, v);
				normalisationValue += v;
			}

			/* Normalise */
			iterator.reset();
			while (iterator.hasNext()) {

				pos = iterator.next();
				exampleDataBuffer.putFloat(pos, (exampleDataBuffer.getFloat(pos) / normalisationValue));
			}

			int tmpindex = 0;
			while (tmpindex < exampleDataBuffer.limit()) {
				tmpindex += 4;
			}

			/* Store current example result, and move on to the next one */
			log.debug(String.format("Put %d bytes to output data buffer", exampleDataBuffer.limit()));
 			outputDataBuffer.put(exampleDataBuffer, 0, exampleDataBuffer.limit(), false);
			offset += length;
		}
		
		batch.setOutput(operator.getId(), outputDataBuffer);
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
