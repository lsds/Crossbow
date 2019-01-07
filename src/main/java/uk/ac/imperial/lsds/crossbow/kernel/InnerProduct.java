package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class InnerProduct extends Kernel {
	
	private final static Logger log = LogManager.getLogger (InnerProduct.class);
	
	private InnerProductConf conf;
	
	LocalVariable _biasmultiplier = null;
	
	public InnerProduct (InnerProductConf conf) {
		this.conf = conf;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 1 input, 1 local variable, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 1, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		Variable []  local = _biasmultiplier.getInitialValue();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		
		/* Setup local variable */
		TheGPU.getInstance().setKernelLocalVariable (id, 0, "biasmultiplier", local[0].getShape().array(), local[0].capacity(), true);
		
		/* Initialise `_biasmultiplier` variable on GPU */
		TheGPU.getInstance().setKernelLocalVariableData (id, 0, local[0].getDataBuffer());
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 3);
		
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0,    "axis", conf.getAxis());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 1, "outputs", conf.numberOfOutputs());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 2,    "bias", conf.hasBias() ? 1 : 0);
		
		return;
	}
	
	public InnerProduct setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		int outputs = conf.numberOfOutputs();
		int axis = conf.getAxis();
		
		/* Dimensions from `axis` onwards are flattened into a  single vector */
		int outer = input.getShape().countElements(0, axis);
		int inner = input.getShape().countElements(axis);
		
		/* Register model variables, "weights" and "bias" */
		
		Variable weights = new Variable ("weights", new Shape (new int [] { outputs, inner }), false);
		weights.initialise (conf.getWeightInitialiser()); // .setValue(1F));
		weights.setLearningRateMultiplier(conf.getWeightsLearningRateMultiplier());
		
		model.register (operator.getId(), weights);
		
		Variable bias = null;
		if (conf.hasBias()) {
			
			bias = new Variable ("bias", new Shape (new int [] { outputs }), false);
			bias.initialise (conf.getBiasInitialiser());
			bias.setLearningRateMultiplier(conf.getBiasLearningRateMultiplier());
			
			model.register(operator.getId(), bias);
		}
		
		/* Configure the output shape */
		
		outputShape = new Shape (axis + 1);
		for (int i = 0; i < axis; ++i)
			outputShape.set(i, inputShape[0].get(i)); // outer
		outputShape.set(axis, outputs); // outputs
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));
		
		Variable var = null;
		if (conf.hasBias()) {
			/*
			 * When multiplied with "bias", the resulting tensor should have the same 
			 * dimensionality as the output.
			 */
			var = new Variable ("bias-multiplier", new Shape (new int [] { outer }), false);
			var.initialise (new InitialiserConf().setValue(1));
			_biasmultiplier = new LocalVariable(var);
			
			log.debug(String.format("Local variable %s", var.getName()));
		}
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? Yes, `weights` and `bias` */
		memoryRequirements.setModelMemoryRequirements (weights.capacity());
		if (conf.hasBias())
			memoryRequirements.incModelMemoryRequirements (bias.capacity());
		
		/* Are there any CPU-specific local variables? Yes, `biasmultiplier` */
		if (conf.hasBias())
			memoryRequirements.setLocalCPUMemoryRequirements (var.capacity());
		
		/* Are there any GPU-specific local variables? Yes, `biasmultiplier` */
		if (conf.hasBias())
			memoryRequirements.setLocalGPUMemoryRequirements (var.capacity());
		
		return this;
	}

	public void compute (Operator[] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		int prev;
		
		/* GEMM variables */
		int M, N, K;
		float alpha, beta;
		int lda, ldb, ldc;
		
		IDataBuffer inputDataBuffer, weightsBuffer;
		int inputStartP, inputEndP;
		int weightStartP, weightEndP;
		
		IDataBuffer outputDataBuffer;
		
		Variable weights, bias;
		
		int axis = conf.getAxis();
		
		Variable [] input  = theInput.get();
		Variable [] output = theOutput.get();

		/* Get input buffer */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();
		inputEndP = getEndPointer ();
	
		/* the output buffer */
		outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		outputDataBuffer.bzero();
		
		model.readLock();
		
		weights = model.getVariable(operator.getId(), 1);
		
		weightsBuffer = weights.getDataBuffer();
		weightStartP = 0;
		weightEndP = weightsBuffer.limit();


		M = input[0].getShape().countElements(0, axis); /* outer   */
		N = conf.numberOfOutputs();                     /* outputs */
		K = input[0].getShape().countElements(axis);    /* inner   */

		alpha = 1F;
		beta  = 0F;

		lda = K; /* if "N", K, else M */
		ldb = K; /* if "N", N, else K */
		ldc = N;
		
		BLAS.getInstance().sgemm("N", "T", 
				M, N, K,
				alpha,
				inputDataBuffer, inputStartP, inputEndP, lda,
				weightsBuffer, weightStartP, weightEndP, ldb,
				beta,
				outputDataBuffer, ldc);
		
		if (conf.hasBias()) {
			
			// M, N, alpha, ldc remain the same
			K = 1;
			lda = K; // if "N", K, else M 
			ldb = N; // if "N", N, else K
			beta = 1F;
			
			Variable [] local = _biasmultiplier.get();
			inputDataBuffer = local[0].getDataBuffer();
			inputStartP = 0;
			inputEndP = inputDataBuffer.limit();
			
			bias = model.getVariable(operator.getId(), 2);
			
			weightsBuffer = bias.getDataBuffer();
			weightStartP = 0;
			weightEndP = weightsBuffer.limit();
			  
			BLAS.getInstance().sgemm("N", "N", 
					M, N, K,
					alpha,
					inputDataBuffer, inputStartP, inputEndP, lda,
					weightsBuffer, weightStartP, weightEndP, ldb,
					beta,
					outputDataBuffer, ldc);
		}
		
		model.readUnlock();
	
		/* Store output in batch for downstream operators */
		batch.setOutput(operator.getId(), outputDataBuffer);
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
