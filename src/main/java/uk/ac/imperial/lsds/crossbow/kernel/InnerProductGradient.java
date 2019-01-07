package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.model.*;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class InnerProductGradient extends Kernel {
	
	private final static Logger log = LogManager.getLogger (InnerProductGradient.class);
	
	private InnerProductConf conf;
	
	private LocalVariable _biasmultiplier;
	
	public InnerProductGradient (InnerProductConf conf) {
		this.conf = conf;
	}

	public InnerProductGradient setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		/* Configure the output shape
		 * 
		 * The output of a gradient operator has the same 
		 * shape as the input of its forward peer.
		 * 
		 * But note that the peer could have more than one inputs.
		 */
		Operator peer = operator.getPeer();
		Shape [] p = peer.getInputShape();
		if (p.length > 1)
			throw new IllegalStateException(String.format("error: peer operator %s has more than one inputs", peer.getName()));
		
		outputShape = p[0].copy();
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));
		
		int outer = output.getShape().countElements(0, conf.getAxis());
		
		Variable var = null;
		if (conf.hasBias()) {
			
			var = new Variable ("bias-multiplier", new Shape (new int [] { outer }), false);
			var.initialise(new InitialiserConf().setValue(1));
			_biasmultiplier = new LocalVariable(var);
			
			log.debug(String.format("Local variable %s", var.getName()));
		}
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? Yes, `biasmultiplier` */
		if (conf.hasBias())
			memoryRequirements.setLocalCPUMemoryRequirements (var.capacity());
		
		/* Are there any GPU-specific local variables? Yes, `biasmultiplier` */
		if (conf.hasBias())
			memoryRequirements.setLocalGPUMemoryRequirements (var.capacity());
		
		return this;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 1 inputs, 1 local variable, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 1, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		Variable [] local =  _biasmultiplier.getInitialValue();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Setup local variable */
		TheGPU.getInstance().setKernelLocalVariable (id, 0, "biasmultiplier", local[0].getShape().array(), local[0].capacity(),  true);
		
		/* Initialise `_biasmultiplier` variable on GPU */
		TheGPU.getInstance().setKernelLocalVariableData (id, 0, local[0].getDataBuffer());
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 3);
		
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0,    "axis", conf.getAxis());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 1, "outputs", conf.numberOfOutputs());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 2,    "bias", conf.hasBias() ? 1 : 0);
		
		return;
	}

	public void compute (Operator[] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		IDataBuffer weightGradientBuffer, biasGradientBuffer;
		
		IDataBuffer inputDataBuffer, peerInputDataBuffer, outputDataBuffer = null;
		int inputStartP, inputEndP, peerInputStartP, peerInputEndP;
		
		Variable [] output;
		int axis;
		
		Variable [] local;
		IDataBuffer multiplier;
		
		int M, N, K;
		float alpha, beta;
		int lda, ldb, ldc;
		
		/* Get the gradient computed by the previous operator */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();
		inputEndP = getEndPointer ();
		
		/* Get the input of peer operator */
		peerInputDataBuffer = getPeerInput (batch, api);
		peerInputStartP = getStartPointer ();
		peerInputEndP = getEndPointer ();
		
		// Compute gradient for weights
		output = theOutput.get();
		axis   = conf.getAxis();
		
		M = output[0].getShape().countElements(0, axis); 	/* outer   */
		N = conf.numberOfOutputs();              			/* outputs */
		K = output[0].getShape().countElements(axis);		/* inner   */

		alpha = 1F;
		beta  = 0F;
		lda = N; 
		ldb = K;
		ldc = K;
		
		/* Prepare buffers for gradients */
		ModelGradient gradient = batch.getModelGradient(model);
        VariableGradient weightsGradient = gradient.getVariableGradient (operator.getPeer().getId(), 1);
		weightGradientBuffer = weightsGradient.getDataBuffer();
		
		/* Reset gradient */
		weightGradientBuffer.bzero();

		/* Compute weight gradient */
		BLAS.getInstance().sgemm ("T", "N", 
				N, K, M,
				alpha,
				inputDataBuffer, inputStartP, inputEndP, lda,
				peerInputDataBuffer,  peerInputStartP,  peerInputEndP,  ldb,
				beta,
				weightGradientBuffer, ldc);
		
		/* Compute bias gradient */
		if (conf.hasBias()) {
			
			VariableGradient biasGradient = null;
			biasGradient = gradient.getVariableGradient (operator.getPeer().getId(), 2);
        	biasGradientBuffer = biasGradient.getDataBuffer();
        	
        	/* Reset gradients */
        	biasGradientBuffer.bzero();
			
			local = _biasmultiplier.get();
			multiplier = local[0].getDataBuffer();

			BLAS.getInstance().sgemv ("T", 
					M, N,
					alpha,
					inputDataBuffer, inputStartP, inputEndP, lda,
					multiplier, 1,
					beta,
					biasGradientBuffer  , 1);
		}

		/* Compute bottom-diff */
		if(! api.isMostUpstream(operator.getPeer())){

			model.readLock();

			Variable weight = model.getVariable(operator.getPeer().getId(), 1);
			IDataBuffer weight_buffer = weight.getDataBuffer();

			int startWeight = 0;
			int endWeight = weight_buffer.limit();

			alpha = 1F;
			beta  = 0F;
			lda = N;
			ldb = K;
			ldc = K;

			/* Get output buffer */
			outputDataBuffer = getCurrentOutput (batch, api);
			output[0].wrap(outputDataBuffer);

			BLAS.getInstance().sgemm ("N", "N",
					M, K, N,
					alpha,
					inputDataBuffer, inputStartP, inputEndP, lda,
					weight_buffer,  startWeight,  endWeight,  ldb,
					beta,
					outputDataBuffer, ldc);

			model.readUnlock();

			/* Store output in batch for downstream operators */
			batch.setOutput(operator.getId(), outputDataBuffer);
		}
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
