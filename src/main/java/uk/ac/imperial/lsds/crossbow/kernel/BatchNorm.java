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
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.BatchNormEstimatedMeanAndVarianceType;
import uk.ac.imperial.lsds.crossbow.types.CudnnKernelType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class BatchNorm extends Kernel {

	private final static Logger log = LogManager.getLogger (BatchNorm.class);

	private BatchNormConf conf;

	/* For the CPU use */
	private LocalVariable spatial_sum_multiplier, num_by_chans, batch_sum_multiplier, temp, x_norm, invVar; 

	/* For the cuDNN use */
	private LocalVariable averageMean, averageVar, newMean, newVar; 
	
	private ThreadLocal<Boolean> isFirstMeanVariance;

	public BatchNorm (BatchNormConf conf) {

		this.conf = conf;
		
		isFirstMeanVariance = new ThreadLocal<Boolean> () {
			protected Boolean initialValue () {
				return new Boolean(true);
			}
		};
	}
	
	@Override
	public IKernel setup (Shape [] inputShape, Model model) {

		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		int axis 		= conf.getAxis (); 						
		int batchsize 	= inputShape[0].countElements(0, axis); 
		int channels 	= inputShape[0].get(axis);
		int spatial_dim = inputShape[0].countElements(axis + 1);
        
		/* Construct local variables */

		Variable var;

		var = new Variable ("spatial_sum_multiplier", new Shape (new int [] { spatial_dim }), false);
		var.initialise (new InitialiserConf().setValue(1));
		spatial_sum_multiplier = new LocalVariable(var);

		var = new Variable ("num_by_chans", new Shape (new int [] { ( channels * batchsize ) }), false);
		var.initialise (new InitialiserConf().setValue(1));
		num_by_chans = new LocalVariable(var);

		var = new Variable ("batch_sum_multiplier", new Shape (new int [] { batchsize }), false);
		var.initialise (new InitialiserConf().setValue(1));
		batch_sum_multiplier = new LocalVariable(var);

		var = new Variable ("temp", inputShape[0], false);
		temp = new LocalVariable(var);

        var = new Variable ("x_norm", inputShape[0], false);
        x_norm = new LocalVariable(var);
        
        var = new Variable ("invVar", new Shape (new int [] { channels }), false);
        invVar = new LocalVariable(var);

		/* Construct local variables for cudnn */

		var = new Variable ("averageMean", new Shape (new int [] { 1, channels, 1, 1 }), false);
		var.initialise (new InitialiserConf().setValue(0)); 
		averageMean = new LocalVariable(var);

		var = new Variable ("averageVar", new Shape (new int [] { 1, channels, 1, 1 }), false);
		var.initialise (new InitialiserConf().setValue(0)); 
		averageVar = new LocalVariable(var);

		var = new Variable ("newMean", new Shape (new int [] { 1, channels, 1, 1 }), false);
        var.initialise (new InitialiserConf().setValue(0));
		newMean = new LocalVariable(var);

		var = new Variable ("newVar", new Shape (new int [] { 1, channels, 1, 1 }), false);
        var.initialise (new InitialiserConf().setValue(0));
		newVar = new LocalVariable(var);

        
		/* Register model variables, "weights" (scale_data) and "bias" (shift_data)*/
		
		Variable weights = new Variable ("weights", new Shape (new int [] { channels }), false);
		weights.initialise (conf.getWeightInitialiser());
		model.register (operator.getId(), weights);
		
		Variable bias = null;
		if (conf.hasBias()) {
			bias = new Variable ("bias", new Shape (new int [] { channels }), false);
			bias.initialise (conf.getBiasInitialiser());
			model.register (operator.getId(), bias);
		}
		
		/* Configure the output shape which is the same as input's shape*/

		outputShape = inputShape[0].copy();

		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);

		log.debug(String.format("Output variable %s", output.getName()));

		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? Yes, `scale` and `bias` */
		memoryRequirements.setModelMemoryRequirements (weights.capacity());
		if (conf.hasBias())
			memoryRequirements.incModelMemoryRequirements (bias.capacity());
		
		/* Are there any CPU-specific local variables? Yes */  
		memoryRequirements.setLocalCPUMemoryRequirements (spatial_sum_multiplier.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (num_by_chans.getInitialValue()[0].capacity()); 
		memoryRequirements.incLocalCPUMemoryRequirements (batch_sum_multiplier.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (temp.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (x_norm.getInitialValue()[0].capacity());
		memoryRequirements.incLocalCPUMemoryRequirements (invVar.getInitialValue()[0].capacity());
		
		
		/* Are there any GPU-specific local variables? Yes */
		memoryRequirements.incLocalGPUMemoryRequirements (newMean.getInitialValue()[0].capacity());
		memoryRequirements.incLocalGPUMemoryRequirements ( newVar.getInitialValue()[0].capacity());
		
		return this;
	}

	@Override
	public void GPURegister () {

		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));

		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 1 input, 2 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 2, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		Variable [] local = averageMean.getInitialValue();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());

		/* Initialise local variables on GPU */
		TheGPU.getInstance().setKernelLocalVariable (id, 0, "newMean",     local[0].getShape().array(), local[0].capacity(), false);
		TheGPU.getInstance().setKernelLocalVariable (id, 1, "newVariance", local[0].getShape().array(), local[0].capacity(), false);
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 4);
		
		TheGPU.getInstance().setKernelConfigurationParameterAsInt	 (id, 0, "globalStatistics", conf.useGlobalStatistics() ? 1 : 0);
		TheGPU.getInstance().setKernelConfigurationParameterAsDouble (id, 1, "epsilon",          conf.getEpsilon());
		TheGPU.getInstance().setKernelConfigurationParameterAsDouble (id, 2, "fraction",         conf.getMovingAverageFraction());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt	 (id, 3, "CMA",             (conf.getEstimatedMeanAndVarianceType() == BatchNormEstimatedMeanAndVarianceType.CMA) ? 1 : 0);
		
		/* Configure cuDNN kernel */
		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.BATCHNORM.getId());
		
		TheGPU.getInstance().cudnnSetBatchNormEstimatedMeanAndVariance (id, local[0].capacity());
		
		int [] dimensions = new int [4];

		input[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);

		output[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelOutputDescriptor	(id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		/* Derived based on input */
		TheGPU.getInstance().cudnnSetBatchNormDescriptor (id);
		
		return;
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {

		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable [] input  =  theInput.get();
		Variable [] output = theOutput.get();
		
		int axis, batchsize, channels, spatial_dim;
		axis 		= conf.getAxis (); 						
		batchsize 	= input[0].getShape().countElements(0, axis); 
		channels 	= input[0].getShape().get(axis);				
		spatial_dim = input[0].getShape().countElements(axis + 1);

		int inputStartP, inputEndP;
        int N;
        float alpha, beta;
        IDataBuffer X, Y;

        IDataBuffer inputDataBuffer, outputDataBuffer;
        IDataBuffer newMeanBuffer, newVarBuffer, averageMeanBuffer, averageVarBuffer;
        IDataBuffer tempBuffer, invVarBuffer, xnormBuffer;
        IDataBuffer weightsBuffer, shiftBuffer;

        int offset;
        IDataBufferIterator j;
      
        /* Get input buffer */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();
		inputEndP = getEndPointer ();

		/* Get the output buffer */
        outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);

		// Find out whether this is TRAINING or TESTING phase
        boolean isTestingPhase = api.isValidationTask();

		if (isTestingPhase) {
			// Use global mean/variance
			
			/* Copy averageMean to newMean and, similarly, averageVar to newVar*/
			averageMeanBuffer 	= averageMean.get()	[0].getDataBuffer();
			newMeanBuffer 		= newMean.get()		[0].getDataBuffer();
			averageVarBuffer 	= averageVar.get()	[0].getDataBuffer();
			newVarBuffer 		= newVar.get()		[0].getDataBuffer();
	        
			copy (averageMeanBuffer, 	newMeanBuffer);
	        copy (averageVarBuffer, 	newVarBuffer);

		} else {

            /* Start by computing mean */
            newMeanBuffer = newMean.get()[0].getDataBuffer();
            compute_mean_per_channel_cpu (batchsize, channels, spatial_dim, inputDataBuffer, newMeanBuffer, inputStartP, inputEndP);
		}

        /* Subtract mean i.e.  Y = X - EX */
		
		// (1.) Copy 'bottom_data' to 'top_data' i.e. from inputbuffer to outputbuffer 
		outputDataBuffer.put(inputDataBuffer, inputStartP, (inputEndP - inputStartP), true);
		// copy (inputbuffer, outputbuffer);
		
        // (2.) Multicast
        // multicast_cpu(N, C, S, mean_.cpu_data(), temp_.mutable_cpu_data()); 
        // Then caffe_cpu_axpby(top_size, Dtype(-1.), temp_.cpu_data(), Dtype(1.), top_data);
        newMeanBuffer   = newMean.get()	[0].getDataBuffer();
        tempBuffer      = temp.get()	[0].getDataBuffer();
        multicast_cpu (batchsize, channels, spatial_dim, newMeanBuffer, tempBuffer, 0, newMeanBuffer.limit());

        N = batchsize * channels * spatial_dim;
        alpha = -1F;
        beta  =  1F;
        X = tempBuffer;
        Y = outputDataBuffer;
        BLAS.getInstance().saxpby( N, alpha, X, 0, X.limit(), 1, beta, Y, 1);

        if (!isTestingPhase) { /* Training */
            
        	/* Compute variance i.e. E (X-EX)^2 , now outputbuffer stores (X - EX) */
        	
        	powx (outputDataBuffer, 2, tempBuffer);
            compute_mean_per_channel_cpu (batchsize, channels, spatial_dim, tempBuffer, newVar.get()[0].getDataBuffer(), 0, tempBuffer.limit());
            
            /* Update global mean and variance */

        	averageMeanBuffer 	= averageMean.get()	[0].getDataBuffer();
        	newVarBuffer 		= newVar.get()		[0].getDataBuffer();
        	averageVarBuffer 	= averageVar.get()	[0].getDataBuffer();
            
        	if (isFirstMeanVariance.get()) {
        		
        		/* If the first-ever new mean and variance are computed, then copy them to the average mean and variance respectively */
            	copy (newMeanBuffer, averageMeanBuffer);
                copy (newVarBuffer , averageVarBuffer);
                
                isFirstMeanVariance.set(false);
    			
            } else {
            	
                /* If this is not the first time new mean and variance are computed, then update the average mean and variance respectively */
        		N = channels;
    			alpha = 1F - (float) conf.getMovingAverageFraction();
    			beta  = 	 (float) conf.getMovingAverageFraction();
    			X = newMeanBuffer;
    			Y = averageMeanBuffer;
    			BLAS.getInstance().saxpby( N, alpha, X, 0, X.limit(), 1, beta, Y, 1);
    			
    			X = newVarBuffer;
    			Y = averageVarBuffer;
    			BLAS.getInstance().saxpby( N, alpha, X, 0, X.limit(), 1, beta, Y, 1);
            }
        }

        /* Derive the inverse Variance; invVar = ( eps+ variance)^(-0.5) */
        
        // caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
        // caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5), variance_.mutable_cpu_data());
        newVarBuffer = newVar.get()[0].getDataBuffer();
        invVarBuffer = invVar.get()[0].getDataBuffer();
        j  = invVarBuffer.getIterator();
        while (j.hasNext()) {
            offset = j.next();
            invVarBuffer.putFloat(offset, (float) Math.pow( newVarBuffer.getFloat(offset) + conf.getEpsilon(), -0.5));
        }

        /* Derive the x_norm; X_norm = (X-EX) * inv_var */
        
        /* Replicate variance to input size */
        multicast_cpu (batchsize, channels, spatial_dim, invVarBuffer, tempBuffer, 0, invVarBuffer.limit());
        
        // Now temp contains inverse variance (different dimensions though)
        mul (outputDataBuffer, tempBuffer, outputDataBuffer);
        
        // Copy from 'top_data' to 'x_norm' to facilitate the backward calculation
        xnormBuffer = x_norm.get()[0].getDataBuffer();
        copy (outputDataBuffer, xnormBuffer);

        /*
         * Scaling phase
         */
        
        /* Y = X_norm * scale[c] + shift[c] */
        
        model.readLock();
        
        weightsBuffer = model.getVariable (operator.getId(), 1).getDataBuffer();
        multicast_cpu (batchsize, channels, spatial_dim, weightsBuffer, tempBuffer, 0, weightsBuffer.limit());
        
        // (1) Y = X_norm * scale[c]  	i.e. outputbuffer[i] * temp[i]
        mul (outputDataBuffer, tempBuffer, outputDataBuffer);
        
        // (2) Y = Y + shift[c] 		i.e. outputbuffer[i] = outputbuffer[i] + shift[i]
        shiftBuffer = model.getVariable (operator.getId(), 2).getDataBuffer();
        multicast_cpu (batchsize, channels, spatial_dim, shiftBuffer, tempBuffer, 0, shiftBuffer.limit());
        // caffe_add(top_size, top_data, temp_.mutable_cpu_data(), top_data);    
        add (outputDataBuffer, tempBuffer, outputDataBuffer);
        
        model.readUnlock();
        
        batch.setOutput (operator.getId(), outputDataBuffer);
	}
	
	public void powx (IDataBuffer src, float p, IDataBuffer dest) {
		
		int offset;
		IDataBufferIterator j = src.getIterator();
        while (j.hasNext()) {
            offset = j.next();
            dest.putFloat (offset, (float) Math.pow (src.getFloat(offset), p));
        }
	}
	
	public void copy (IDataBuffer src, IDataBuffer dest) {
		
		int offset;
		IDataBufferIterator j = src.getIterator();
        while (j.hasNext()) {
            offset = j.next();
            dest.putFloat (offset, src.getFloat(offset));
        }
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

    public LocalVariable getSaveMean () {
        return newMean;
    }
    
    public LocalVariable getSaveVar () {
        return newVar;
    }

    public LocalVariable getXNorm () {
        return x_norm;
    }

    public LocalVariable getInvVar () {
        return invVar;
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
