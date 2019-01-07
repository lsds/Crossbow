package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConvConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.CudnnKernelType;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class Conv extends Kernel {
	
	private final static Logger log = LogManager.getLogger (Conv.class);
	
	private final static void checkSpatialDimensions (int x, int y) {
		
		if ((x != 0) && (x != 1) && (x != y))
			throw new IllegalArgumentException ("error: invalid number of spatial dimensions");
	}
	
	private ConvConf conf;
	
	private int spatialDimensions;
	
	Shape kernel, stride, padding;
	Shape image;
	
	Shape weightShape, biasShape = null;
	
	boolean scalar;
	
	LocalVariable _column, _biasmultiplier;
	
	public Conv (ConvConf conf) {
		
		this.conf = conf;
	}
	
	public Conv setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		int axis = conf.getAxis();
		
		/* The first spatial axis is `axis + 1` */
		spatialDimensions = inputShape[0].dimensions() - (axis + 1);
		
		if (spatialDimensions  < 0)
			throw new IllegalArgumentException ("error: invalid number of spatial dimensions");
		else
		if (spatialDimensions == 0) {
			log.warn("Number of spatial dimensions is 0. Setting to 1...");
			spatialDimensions = Math.max(spatialDimensions, 1);
		}
		
		/* Setup kernel, stride, and padding */
		
		checkSpatialDimensions (conf.getKernelSize (), spatialDimensions);
		checkSpatialDimensions (conf.getStrideSize (), spatialDimensions);
		checkSpatialDimensions (conf.getPaddingSize(), spatialDimensions);
		
		kernel  = new Shape (spatialDimensions);
		stride  = new Shape (spatialDimensions);
		padding = new Shape (spatialDimensions);
		
		for (int i = 0; i < spatialDimensions; ++i) {
			
			 kernel.set (i, (conf.getKernelSize  () == 0) ? 1 : conf.getKernel  ((conf.getKernelSize  () == 1) ? 0 : i));
			 stride.set (i, (conf.getStrideSize  () == 0) ? 1 : conf.getStride  ((conf.getStrideSize  () == 1) ? 0 : i));
			padding.set (i, (conf.getPaddingSize () == 0) ? 0 : conf.getPadding ((conf.getPaddingSize () == 1) ? 0 : i));
		}
		
		log.debug(String.format("Kernel  shaped as %s",  kernel.toString()));
		log.debug(String.format("Stride  shaped as %s",  stride.toString()));
		log.debug(String.format("Padding shaped as %s", padding.toString()));
		
		log.debug("Kernel:" + kernel.get(0) + "," + kernel.get(1));
		log.debug("Stride:" + stride.get(0) + "," + stride.get(1));
		log.debug("Padding:" + padding.get(0) + "," + padding.get(1));
		
		/* Check if convolution kernel is scalar (1 x 1) */
		scalar = true;
		for (int i = 0; i < spatialDimensions; ++i)
			scalar &= (kernel.get(i) == 1 && stride.get(i) == 1 && padding.get(i) == 0);
		
		/* Register model variables, "weights" and "bias" */
		
		int groups = conf.numberOfGroups(); /* Default value is 1 */
		if(groups > 1)
			throw new UnsupportedOperationException ("error: number of groups more than 1 is not yet supported.");
		
		int channels = inputShape[0].get(axis);
		log.debug(String.format("%d channels, %d groups", channels, groups));
		
		if ((channels % groups) != 0)
			throw new IllegalArgumentException ("error: number of channels must be a multiple of group size");
		
		int outputs = conf.numberOfOutputs (); /* The number of filters */
		if ((outputs  % groups) != 0)
			throw new IllegalArgumentException ("error: number of outputs must be a multiple of group size");
		
		/* Shape weights */
		Shape w = new Shape (2);
		
		w.set(0,  outputs);
		w.set(1, channels / groups);
		for (int i = 0; i < spatialDimensions; i++) 
			w.push (kernel.get(i));
		
		Variable weights = new Variable ("weights", w, false);
		
		weights.initialise (conf.getWeightInitialiser());
		weights.setLearningRateMultiplier(conf.getWeightsLearningRateMultiplier());
		
		model.register (operator.getId(), weights);
		
		weightShape = new Shape (w.array());
		
		Variable bias = null;
		
		if (conf.hasBias()) {
			
			bias = new Variable ("bias", new Shape (new int [] { outputs }), false);
			bias.initialise (conf.getBiasInitialiser());
			bias.setLearningRateMultiplier(conf.getBiasLearningRateMultiplier());
			
			model.register(operator.getId(), bias);
			
			biasShape = new Shape (new int [] { outputs });
		}
		
		/* Compute output shape */
		
		outputShape = new Shape ();
		
		for (int i = 0; i < axis; ++i)
			outputShape.push(inputShape[0].get(i)); // Mini-batch size
		
		outputShape.push(outputs); // The number of filters
		
		for (int i = 0; i < spatialDimensions; ++i) {
			int d = (inputShape[0].get(axis + i + 1) + 2 * padding.get(i) - kernel.get(i)) / stride.get(i) + 1;
			outputShape.push(d); // Output size
		}
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));
		
		image = new Shape(spatialDimensions + 1);
		for (int i = 0; i < spatialDimensions + 1; ++i)
			image.set(i, inputShape[0].get(axis + i));
		
		/* Configure local variable */
		
		int kernelSpatialDimensions = w.countElements(1);
		
		Shape columnShape = new Shape (spatialDimensions + 1); // spatialDimensions = 2
		
		columnShape.set (0, kernelSpatialDimensions * groups);
		
		for (int i = 0; i < spatialDimensions; ++i) {
			int d = (inputShape[0].get(axis + i + 1) + 2 * padding.get(i) - kernel.get(i)) / stride.get(i) + 1;
			columnShape.set (i + 1, d); // d = weight or height of the output
		}
		
		Variable column = new Variable ("column", columnShape, false);
		column.initialise (new InitialiserConf().setValue(0));
		
		_column = new LocalVariable(column);
		
		log.debug(String.format("Local variable %s", column.getName()));
		
		Variable biasmultiplier = null;
		if (conf.hasBias()) {
			/*
			 * When multiplied with "bias", the resulting variable should have the same 
			 * dimensionality as the output.
			 */
			biasmultiplier = new Variable ("bias-multiplier", new Shape (new int [] { outputShape.countElements(axis + 1) }), false);
			biasmultiplier.initialise (new InitialiserConf().setValue(1));
			
			_biasmultiplier = new LocalVariable(biasmultiplier);
			
			log.debug(String.format("Local variable %s", biasmultiplier.getName()));
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
		
		/* Are there any CPU-specific local variables? Yes, `column` and `biasmultiplier` */
		memoryRequirements.setLocalCPUMemoryRequirements (column.capacity());
		if (conf.hasBias())
			memoryRequirements.incLocalCPUMemoryRequirements (biasmultiplier.capacity());
		
		/* Are there any GPU-specific local variables? Yes, but they cannot be defined at the moment */
		memoryRequirements.setLocalGPUMemoryRequirements (0);
		
		return this;
	}
	
	public void GPURegister () {
		
		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
		
		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		/* 1 input, 3 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 3, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Set GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "bias", conf.hasBias() ? 1 : 0);
		
		/* Set cuDNN kernel */
		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.CONV.getId());
		
		int [] dimensions = new int [4];
			
		input[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
			
		output[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		TheGPU.getInstance().cudnnSetConvolutionDescriptor(id, padding.get(0), padding.get(1), stride.get(0), stride.get(1));
		
		weightShape.getNCHW(dimensions);
		TheGPU.getInstance().cudnnSetConvolutionFilterDescriptor(id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		if (conf.hasBias())
			TheGPU.getInstance().cudnnSetConvolutionBiasDescriptor(id, 1, biasShape.get(0), 1, 1);
		
		/* Change -1 (unlimited memory) to 0 to eliminate workspace memory requirements on GPU */
		
		double __threshold = -1;
		
		int __limit_fwd = -1;
		int __limit_bwd = -1;
		
		int forwardWorkspaceSizeInBytes        = TheGPU.getInstance().cudnnConfigureConvolutionForwardAlgorithm (id, __limit_fwd, __threshold); 
		int backwardFilterWorkspaceSizeInBytes = TheGPU.getInstance().cudnnConfigureConvolutionBackwardFilterAlgorithm (id, __limit_bwd, __threshold);
		int backwardDataWorkspaceSizeInBytes   = TheGPU.getInstance().cudnnConfigureConvolutionBackwardDataAlgorithm (id, __limit_bwd, __threshold);
		
		log.debug (String.format("Forward workspace size is %d bytes", forwardWorkspaceSizeInBytes));
		log.debug (String.format("Backward filter workspace size is %d bytes", backwardFilterWorkspaceSizeInBytes));
		log.debug (String.format("Backward data workspace size is %d bytes", backwardDataWorkspaceSizeInBytes));
		
		
		memoryRequirements.setLocalGPUMemoryRequirements
			((long) forwardWorkspaceSizeInBytes + (long) backwardFilterWorkspaceSizeInBytes + (long) backwardDataWorkspaceSizeInBytes);
		
		/* Set local variables */
		if (forwardWorkspaceSizeInBytes > 0) {
			TheGPU.getInstance().setKernelLocalVariable (id, 0, "forwardWorkSpace", 
					new int [] { (forwardWorkspaceSizeInBytes / 4) }, forwardWorkspaceSizeInBytes, false);
		}
		
		if (backwardFilterWorkspaceSizeInBytes > 0) {
			TheGPU.getInstance().setKernelLocalVariable (id, 1, "backwardFilterWorkSpace", 
					new int [] { (backwardFilterWorkspaceSizeInBytes / 4) }, backwardFilterWorkspaceSizeInBytes, false);
		}
		
		if (backwardDataWorkspaceSizeInBytes > 0) {
			TheGPU.getInstance().setKernelLocalVariable (id, 2, "backwardDataWorkSpace", 
					new int [] { (backwardDataWorkspaceSizeInBytes / 4) }, backwardDataWorkspaceSizeInBytes, false);
		}
		
		return;
	}
	
	private void imageToColumn2D (IDataBuffer image, int offset, IDataBuffer column, int channels,
		
		int   imageHeight, int   imageWidth, 
		int  kernelHeight, int  kernelWidth,
		int paddingHeight, int paddingWidth,
		int  strideHeight, int  strideWidth
		
		) {
		
		int output_h  = (imageHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
		int output_w  = (imageWidth  + 2 * paddingWidth  - kernelWidth)  / strideWidth  + 1;
		
		int entirecolumn = channels * kernelHeight * kernelWidth;
		
		for (int columnelement = 0; columnelement < entirecolumn; ++columnelement) {
			
			int  widthOffset =  columnelement % kernelWidth;
			int heightOffset = (columnelement / kernelWidth) % kernelHeight;
			
			int c_ = columnelement / kernelHeight / kernelWidth;
			
			for (int h = 0; h < output_h; ++h) {
				
				for (int w = 0; w < output_w; ++w) {
					
					int h_ = h * strideHeight - paddingHeight + heightOffset;
					int w_ = w * strideWidth  - paddingWidth  +  widthOffset;
					
					int index = ((columnelement * output_h + h) * output_w + w) * DataType.FLOAT.sizeOf();
					float value = 0F;
					if (h_ >= 0 && w_ >= 0 && h_ < imageHeight && w_ < imageWidth) {
						value = image.getFloat(offset + (((c_ * imageHeight + h_) * imageWidth + w_) * DataType.FLOAT.sizeOf()));
					}

					column.putFloat(index, value);
				}
			}
		}
	}
	
	private void imageToColumn (IDataBuffer inputbuffer, int offset, IDataBuffer columnbuffer, int channels) {
		
		if (spatialDimensions != 2)
			throw new UnsupportedOperationException("error: multi-dimensional convolution is not yet supported");
		
		imageToColumn2D (inputbuffer, offset, columnbuffer, channels,
				
			  image.get(1),   image.get(2), /* Image  height & width */
			 kernel.get(0),  kernel.get(1), /* Kernel height & width */ 
			padding.get(0), padding.get(1), /* ...                   */
			 stride.get(0),  stride.get(1)
		);
	}
	
	public void compute (Operator[] previous, Batch batch, Model model, ITask api) {

		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		int axis = conf.getAxis();
		int groups = conf.numberOfGroups();
		int outputs = conf.numberOfOutputs();
		
		/* Get thread-local variables */
		Variable [] input  =  theInput.get();
		Variable [] output = theOutput.get();
		
		int channels = input[0].getShape().get(axis);
		
		/* Get input buffer */
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		int inputEndP = getEndPointer ();
		
		/* Get an output buffer */
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		
		int  inputoffset,  inputvectorsize;
		int outputoffset, outputvectorsize;
		/*
		 * An image has one of more color channels `c` (typically, c = 1 for grayscale and c = 3 for RGB images),
		 * a height `h` and a width `w`. 
		 * 
		 * Convolution operators ignore this spatial structure of images and rather treat them as one vector with
		 * dimensions `chw`.
		 */
		inputvectorsize  =  input[0].getShape().countElements(axis) *  input[0].getType().sizeOf(); //Size of each image input
		outputvectorsize = output[0].getShape().countElements(axis) * output[0].getType().sizeOf(); //Size of each output
		
		int batchsize = input[0].getShape().countElements(0, axis);
		
		Variable [] column = _column.get();
		IDataBuffer columnBuffer = column[0].getDataBuffer(); //For matmul operation
		
		model.readLock();
		
		Variable weights = model.getVariable(operator.getId(), 1);
		log.debug("Weight checksum is " + weights.computeChecksum());
		IDataBuffer weightsBuffer = weights.getDataBuffer();
		
		IDataBuffer biasBuffer = null, biasmultiplierBuffer = null;
		
		if (conf.hasBias()) {
			Variable biasVar = model.getVariable(operator.getId(), 2);
			log.debug("Bias checksum is " + biasVar.computeChecksum());
			biasBuffer = model.getVariable(operator.getId(), 2).getDataBuffer();
			biasmultiplierBuffer = _biasmultiplier.get()[0].getDataBuffer();
		}
		
		/* GEMM helper variables */
		int M = outputs / groups;
		int N = output[0].getShape().countElements(axis + 1);
		int K = weights.getShape().countElements(1);
		
		/*
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		int ldc = N;
		*/  
		int lda = K;
		int ldb = N;
		int ldc = N;
		
		int Alimit = M * K * 4;
		int Blimit = K * N * 4;
		int Climit = M * N * 4;
		
		int weightsoffset = M * K * weights.getType().sizeOf();
		int columnoffset  = N * K *  column[0].getType().sizeOf();
		
		int output_group_offset = M * N;
		
		int M2 = outputs;
		int N2 = output[0].getShape().countElements(axis + 1);
		int K2 = 1;
		int lda2 = K2;
		int ldb2 = N2;
		int ldc2 = N2;
		
		int Climit2 = M2 * N2 * 4;
		
		for (int n = 0; n < batchsize; ++n) { //For each training example
			
			 inputoffset = n *  inputvectorsize;
			outputoffset = n * outputvectorsize;
			
			// System.out.println(String.format("[DBG] n = %d input offset %d output offset %d", n, inputoffset / 4, outputoffset / 4));
			
			if (! scalar) { /* Not a (1 x 1) convolution */
				
				imageToColumn (inputDataBuffer, (inputStartP + inputoffset), columnBuffer, channels);
				
				for (int g = 0; g < groups; ++g) {
					
					BLAS.getInstance().sgemm ("N", "N", 
						M, N, K,
						1F, 
						weightsBuffer, g * weightsoffset, g * weightsoffset + Alimit, lda, /* A */
						columnBuffer , g *  columnoffset, g *  columnoffset + Blimit, ldb, /* B */
						0F, 
						outputDataBuffer , (outputoffset + g * output_group_offset), (outputoffset + g *  output_group_offset) + Climit, ldc); /* C */
				}
				
			} else {
				
				for (int g = 0; g < groups; ++g) {
					
					BLAS.getInstance().sgemm ("N", "N", 
						M, N, K,
						1F, 
						weightsBuffer, g *  weightsoffset                        ,   g * weightsoffset + Alimit, lda,
						inputDataBuffer  , g *  columnoffset + (inputStartP + inputoffset),   g *  columnoffset + (inputStartP + inputoffset) + Blimit, ldb,
						0F, 
						outputDataBuffer , g *  outputoffset                         ,   g *  outputoffset + Climit, ldc);
				}
			}
			
			if (conf.hasBias()) {
				
				/* forward_cpu_bias (output + n * top_dim_, bias) */
				BLAS.getInstance().sgemm (
					"N", "N",
					M2,N2,K2,
					1F, 
					biasBuffer,           0, biasBuffer.limit(), lda2,
					biasmultiplierBuffer, 0, biasmultiplierBuffer.limit(), ldb2,
					1F, 
					outputDataBuffer, outputoffset, outputoffset + Climit2, ldc2);
			}
		}
		
		model.readUnlock();
		
		/* Store output in batch for downstream operators */
		batch.setOutput(operator.getId(), outputDataBuffer);
	}
	
	public LocalVariable getLocalVariableColumn (){
        return _column;
    }

    public Shape getImageShape (){
        return image;
    }

    public boolean getScalar (){
        return scalar;
    }
    
    public LocalVariable getBiasMultiplier () {
        return _biasmultiplier;
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
