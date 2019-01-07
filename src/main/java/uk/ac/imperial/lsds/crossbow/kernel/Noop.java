package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.NoopConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Noop extends Kernel {

	private final static Logger log = LogManager.getLogger (Noop.class);

	private NoopConf conf;

	public Noop (NoopConf conf) {
		this.conf = conf;
	}

	public Noop setup (Shape [] inputShape, Model model) {

		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));

		/* Model variable is the identity matrix of the input */
		int axis = conf.getAxis();
		int N = inputShape[0].countElements(axis);
		Variable var = new Variable ("unit", new Shape (new int [] { N, N }), false);
		/* Initialise unit array */
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				int offset = ((i * N) + j) * var.getType().sizeOf();
				if (i == j) {
					var.getDataBuffer().putFloat(offset, 1);
				}
				else {
					var.getDataBuffer().putFloat(offset, 0);
				}
			}
		}
		/* Register model variable */
		model.register (operator.getId(), var);
		
		/* Configure the output shape */

		outputShape = inputShape[0].copy();

		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);

		log.debug(String.format("Output variable %s", output.getName()));
		
		/* Set memory requirements */
		
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		memoryRequirements.setModelMemoryRequirements(var.capacity());
		
		return this;
	}

	public void GPURegister () {

		int id = operator.getId();
		String name = this.getClass().getSimpleName();

		/* 1 input, 0 local variables, 1 output, pull = false */
		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, true); // (isLossKernel() || isAccuracyKernel()));

		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();

		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis", conf.getAxis());

		return;
	}
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		Variable []  input =  theInput.get();
		Variable [] output = theOutput.get();
		
		/* Get first model variable */
		Variable var = model.getVariable(operator.getId(), 1);
		
		if (log.isDebugEnabled()) {

			log.debug(String.format("Input  variable shape is %s",  input[0].getShape()));
			log.debug(String.format("Output variable shape is %s", output[0].getShape()));
			log.debug(String.format("Model  variable shape is %s",       var.getShape()));
		}
		
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		int inputEndP = getEndPointer ();
		
		/*
		IDataBuffer inputDataBuffer = batch.getInputBuffer(0);
		int inputStartP = batch.getBufferStartPointer(0);
		int inputEndP = batch.getBufferEndPointer(0);
		*/
		
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer); 
		
		model.readLock();
		
		IDataBuffer modelDataBuffer = var.getDataBuffer();
		
		/* Multiply tensors (output = input x model = input) */
		int M = input[0].getShape().countElements(0, conf.getAxis());
		int K = input[0].getShape().countElements(conf.getAxis());
		int N = K;
		float alpha = 1F;
		int lda = K;
		int ldb = N;
		float beta = 0F;
		int ldc = N;
		
		BLAS.getInstance().sgemm("N", "N", 
				M, N, K, 
				alpha, 
				inputDataBuffer, inputStartP, inputEndP, lda, 
				modelDataBuffer, 0, modelDataBuffer.limit(), ldb, 
				beta, 
				outputDataBuffer, ldc);
		
		model.readUnlock();
		
		/*
		log.debug(String.format("Batch %d's output checksum is %.5f", batch.getId(), outputDataBuffer.computeChecksum()));
		log.debug(String.format("Batch %d's model  checksum is %.5f", batch.getId(),  modelDataBuffer.computeChecksum()));
		*/
		
		/* Is output equal to input? */
		int offset;
		IDataBufferIterator j = outputDataBuffer.getIterator();
        while (j.hasNext()) {
            offset = j.next();
            if (outputDataBuffer.getFloat(offset) != inputDataBuffer.getFloat(inputStartP + offset)) {
            	throw new RuntimeException (String.format("error: invalid output (at batch %d offset %d)", batch.getId(), offset));
            }
        }
		
		batch.setOutput (operator.getId(), outputDataBuffer);
	}
	
	public ModelAccess getModelAccessType () {
		return ModelAccess.RO;
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
