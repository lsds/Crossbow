package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.BatchNormConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.ModelGradient;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.model.VariableGradient;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class BatchNormGradient extends Kernel {
	
	private final static Logger log = LogManager.getLogger (BatchNormGradient.class);
	
	private BatchNormConf conf;
	
	/* For the CPU use */
	private LocalVariable spatial_sum_multiplier, num_by_chans, batch_sum_multiplier, temp, x_norm_diff, temp_C;
	
	public BatchNormGradient (BatchNormConf conf) {
		
		this.conf = conf;
	}
	
	@Override
	public IKernel setup (Shape[] inputShape, Model model) {
		
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
		
		int axis = conf.getAxis();
		int batchsize = inputShape[0].countElements(0, axis);
		int channels = inputShape[0].get(axis);
		int spatial_dim = inputShape[0].countElements(axis + 1);
		
		/* Construct local variables */

		Variable var;

        var = new Variable ("spatial_sum_multiplier", new Shape (new int [] { spatial_dim }), false);
        var.initialise (new InitialiserConf().setValue(1));
        spatial_sum_multiplier = new LocalVariable(var);

        var = new Variable ("num_by_chans", new Shape (new int [] { ( channels * batchsize ) }), false);
        var.initialise (new InitialiserConf().setValue(1));
        num_by_chans = new LocalVariable(var);

        var = new Variable ("batch_sum_multiplier",   new Shape (new int [] { batchsize }), false);
        var.initialise (new InitialiserConf().setValue(1)); 
        batch_sum_multiplier = new LocalVariable(var);

        var = new Variable ("temp", inputShape[0], false);
        temp = new LocalVariable(var);

        var = new Variable ("xnormdiff", inputShape[0], false); //TODO : Check if we can optimise memory usage by removing this variable
        x_norm_diff = new LocalVariable(var);

        var = new Variable ("temp_C", new Shape (new int [] { channels }), false);
        temp_C = new LocalVariable(var);
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? Yes */  
		memoryRequirements.setLocalCPUMemoryRequirements (spatial_sum_multiplier.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (num_by_chans.getInitialValue()[0].capacity()); 
		memoryRequirements.incLocalCPUMemoryRequirements (batch_sum_multiplier.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (temp.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (x_norm_diff.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (temp_C.getInitialValue()[0].capacity());
		
		/* Are there any GPU-specific local variables?  No */
		memoryRequirements.setLocalGPUMemoryRequirements (0); 

		return this;
	}

	@Override
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
		TheGPU.getInstance().setKernelConfigurationParameterAsDouble (id, 0, "epsilon", conf.getEpsilon());
		
		return;
	}
	
	@Override
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {

        if (previous != null && previous.length > 1)
            throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

        Variable [] input  =  theInput.get(); 
        Variable [] output = theOutput.get();

        int inputStartP;
        int N;
        float alpha, beta;
        IDataBuffer  X, Y;

        int axis, batchsize, channels, spatial_dim;
        axis = conf.getAxis();
		batchsize = input[0].getShape().countElements(0, axis);
		channels = input[0].getShape().get(axis);
		spatial_dim = input[0].getShape().countElements(axis + 1);

        IDataBuffer inputDataBuffer, outputDataBuffer;
        IDataBuffer weightGradientBuffer, biasGradientBuffer; 
        IDataBuffer weightsBuffer; 
        IDataBuffer topDataBuffer, topDiffBuffer, bottomDiffBuffer;

        LocalVariable x_norm;
        IDataBuffer xnormBuffer, tempBuffer, xnormDiffBuffer, temp_C_Buffer;

        /* Get input buffer */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();

        /* Get the output buffer */
        outputDataBuffer = getCurrentOutput (batch, api);
        output[0].wrap(outputDataBuffer);

        /* Get the peer */ 
        ModelGradient gradient = batch.getModelGradient(model);
        VariableGradient weightsGradient = gradient.getVariableGradient (operator.getPeer().getId(), 1);
		weightGradientBuffer = weightsGradient.getDataBuffer();
		weightGradientBuffer.bzero();

        /* Get 'x_norm' variable from the peer */
        x_norm = ((BatchNorm) operator.getPeer().getKernel()).getXNorm();
        xnormBuffer     = x_norm.get()      [0].getDataBuffer();
        xnormDiffBuffer = x_norm_diff.get() [0].getDataBuffer();

        /* Stage 1: Compute dE/d(scale) and dE/d(shift) */

        // (1) scaleDiffBuffer: dE/d(scale)  =  sum(dE/dY .* X_norm)
        tempBuffer = temp.get()[0].getDataBuffer();
        mul (inputDataBuffer, xnormBuffer, tempBuffer);
        compute_sum_per_channel_cpu (batchsize, channels, spatial_dim, tempBuffer, weightGradientBuffer, 0, tempBuffer.limit());

        // (2) shiftDiffBuffer: dE/d(shift) = sum (dE/dY)
        VariableGradient biasGradient = null;
        if (conf.hasBias()) {

        	biasGradient = gradient.getVariableGradient (operator.getPeer().getId(), 2);
        	biasGradientBuffer = biasGradient.getDataBuffer();
        	
        	/* Reset gradients */
            biasGradientBuffer.bzero();

            compute_sum_per_channel_cpu (batchsize, channels, spatial_dim, inputDataBuffer, biasGradientBuffer, inputStartP, inputDataBuffer.limit());
        }

        /* Stage 2: Backprop dE/d(x_norm) = dE/dY .* scale */

        // dE/d(X_norm) = dE/dY * scale[c]
        model.readLock();
        
        Variable weight = model.getVariable(operator.getPeer().getId(), 1);
        weightsBuffer = weight.getDataBuffer();
        multicast_cpu (batchsize, channels, spatial_dim, weightsBuffer, tempBuffer, 0, weightsBuffer.limit());
        model.readUnlock();
        
        mul (inputDataBuffer, tempBuffer, xnormDiffBuffer);

        /* Stage 3: backprop dE/dY --> dE/dX */
        // i.e. dE(Y)/dX =  (dE/dY - mean(dE/dY) - mean(dE/dY .* Y) .* Y) ./ sqrt(var(X) + eps)

        // From here, xnormBuffer represents top_data and xnormDiffBuffer represents top_diff for BN
        topDataBuffer = xnormBuffer;
        topDiffBuffer = xnormDiffBuffer;
        bottomDiffBuffer = outputDataBuffer;

        // temp = mean(dE/dY .* Y)
        mul (topDiffBuffer, topDataBuffer, tempBuffer);
        temp_C_Buffer = temp_C.get()[0].getDataBuffer();
        compute_mean_per_channel_cpu (batchsize, channels, spatial_dim, tempBuffer, temp_C_Buffer, 0, tempBuffer.limit());
        multicast_cpu (batchsize, channels, spatial_dim, temp_C_Buffer, tempBuffer, 0, temp_C_Buffer.limit());

        // bottom = mean(dE/dY .* Y) .* Y
        mul (tempBuffer, topDataBuffer, bottomDiffBuffer);

        // temp = mean(dE/dY)
        compute_mean_per_channel_cpu (batchsize, channels, spatial_dim, topDiffBuffer, temp_C_Buffer, 0, topDiffBuffer.limit());
        multicast_cpu (batchsize, channels, spatial_dim, temp_C_Buffer, tempBuffer, 0, temp_C_Buffer.limit());

        // bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
        add (tempBuffer, bottomDiffBuffer, bottomDiffBuffer);

        // bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
        // caffe_cpu_axpby(top_size (N), Dtype(1.) (alpha), top_diff (X), Dtype(-1.) (beta), bottom_diff (Y));
        N = batchsize * channels * spatial_dim;
        alpha = 1F;
        beta = -1F;
        X = topDiffBuffer;
        Y = bottomDiffBuffer;
        int startTopDiff = 0;
        BLAS.getInstance().saxpby( N, alpha, X, startTopDiff, X.limit(), 1, beta, Y, 1);


        // dE/dX = dE/dX ./ sqrt(var(X) + eps)
        LocalVariable invVar 		= ((BatchNorm) operator.getPeer().getKernel()).getInvVar();
        IDataBuffer invVarBuffer 	= invVar.get()[0].getDataBuffer();
        multicast_cpu (batchsize, channels, spatial_dim, invVarBuffer, tempBuffer, 0, invVarBuffer.limit());
        mul (bottomDiffBuffer, tempBuffer, bottomDiffBuffer);


        batch.setOutput(operator.getId(), bottomDiffBuffer);
	}

    public void mul (IDataBuffer a, IDataBuffer b, IDataBuffer y) {

        IDataBufferIterator j  = a.getIterator();
        int offset;
        while (j.hasNext()) {
            offset = j.next();
            y.putFloat (offset, a.getFloat(offset) * b.getFloat(offset));
        }
    }

    public void add (IDataBuffer a, IDataBuffer b, IDataBuffer y) {

        IDataBufferIterator j  = a.getIterator();
        int offset;
        while (j.hasNext()) {
            offset = j.next();
            y.putFloat (offset, a.getFloat(offset) + b.getFloat(offset));
        }
    }

    public void compute_mean_per_channel_cpu (int batchsize, int channels, int spatial_dim, IDataBuffer A, IDataBuffer _Y, int startA, int endA) {

        compute_sum_per_channel_cpu (batchsize, channels, spatial_dim, A, _Y, startA, endA);

        // Scale newMean with 1F / (batchsize * spatial_dim)
        int N = channels;
        float alpha = 0F;
        float beta = 1F / (batchsize * spatial_dim);
        IDataBuffer X = _Y; // TODO: Can we do better?
        IDataBuffer Y = _Y;

        BLAS.getInstance().saxpby( N, alpha, X, 0, X.limit(), 1, beta, Y, 1);
    }

    public void compute_sum_per_channel_cpu (int batchsize, int channels, int spatial_dim, IDataBuffer _A, IDataBuffer _Y, int startA, int endA) {

        int M, N;
        float alpha, beta;
        int lda;
        IDataBuffer A, X, Y;

        M = channels * batchsize;
        N = spatial_dim;
        alpha = 1F; // 1F / (batchsize * spatial_dim);
        beta = 0F;
        lda = N;
        A = _A;
        X = spatial_sum_multiplier.get()[0].getDataBuffer();
        Y = num_by_chans.get()[0].getDataBuffer();

        BLAS.getInstance().sgemv ("N",
                M, N,
                alpha,
                A, startA, endA, lda,
                X, 1,
                beta,
                Y, 1);

        M = batchsize;
        N = channels;
        alpha = 1F;
        beta = 0F;
        lda = N;
        A = num_by_chans.get()[0].getDataBuffer();
        X = batch_sum_multiplier.get()[0].getDataBuffer();
        Y = _Y;

        BLAS.getInstance().sgemv ("T",
                M, N,
                alpha,
                A, 0, A.limit(), lda,
                X, 1,
                beta,
                Y, 1);
    }

    public void multicast_cpu (int batchsize, int channels, int spatial_dim, IDataBuffer X, IDataBuffer Y, int startX, int endX) {

        int startA, endA, startB, endB;
        int M, N, K;
        float alpha, beta;
        int lda, ldb, ldc;
        IDataBuffer A, B, C;

        M = batchsize ;
        N = channels;
        K = 1;
        alpha = 1F;
        beta = 0F;
        lda = K; /* if "N", K, else M */
        ldb = N; /* if "N", N, else K */
        ldc = N;
        A = batch_sum_multiplier.get()[0].getDataBuffer();
        B = X;
        C = num_by_chans.get()[0].getDataBuffer();
        startA = 0;
        endA = A.limit();
        startB = startX;
        endB = endX;

        BLAS.getInstance().sgemm("N", "N",
                M, N, K,
                alpha,
                A, startA, endA, lda,
                B, startB, endB, ldb,
                beta,
                C, ldc);

        M = batchsize * channels ;
        N = spatial_dim;
        K = 1;
        alpha = 1F;
        beta = 0F;
        lda = K; /* if "N", K, else M */
        ldb = N; /* if "N", N, else K */
        ldc = N;
        A = num_by_chans.get()[0].getDataBuffer();
        B = spatial_sum_multiplier.get()[0].getDataBuffer();
        C = Y;
        startA = 0;
        endA = A.limit();
        startB = 0;
        endB = B.limit();

        BLAS.getInstance().sgemm("N", "N",
                M, N, K,
                alpha,
                A, startA, endA, lda,
                B, startB, endB, ldb,
                beta,
                C, ldc);
    }

	public ModelAccess getModelAccessType () {
		return ModelAccess.RW;
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
