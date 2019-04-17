package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConvConf;
import uk.ac.imperial.lsds.crossbow.model.*;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public class ConvGradient extends Kernel {

	private final static Logger log = LogManager.getLogger (ConvGradient.class);

	ConvConf conf;

    private int spatialDimensions;

    Shape kernel, stride, padding;
    Shape imageShape;

	LocalVariable columns, _biasmultiplier;
	
	public ConvGradient (ConvConf conf) {
		
		this.conf = conf;
	}

	public ConvGradient setup (Shape [] inputShape, Model model) {

		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

		Variable input = new Variable ("input", inputShape[0], true); //Top_diff
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

		Variable output = new Variable ("output", outputShape, true); //Bottom_diff
		theOutput = new LocalVariable (output);

		log.debug(String.format("Output variable %s", output.getName()));

        int axis = conf.getAxis();
        spatialDimensions = inputShape[0].dimensions() - (axis + 1);

        if (spatialDimensions  < 0)
            throw new IllegalArgumentException ("error: invalid number of spatial dimensions");
        else
        if (spatialDimensions == 0) {
            log.warn("Number of spatial dimensions is 0. Setting to 1...");
            spatialDimensions = Math.max(spatialDimensions, 1);
        }

        kernel  = new Shape (spatialDimensions);
        stride  = new Shape (spatialDimensions);
        padding = new Shape (spatialDimensions);

        for (int i = 0; i < spatialDimensions; ++i) {

            kernel.set (i, (conf.getKernelSize  () == 0) ? 1 : conf.getKernel  ((conf.getKernelSize  () == 1) ? 0 : i));
            stride.set (i, (conf.getStrideSize  () == 0) ? 1 : conf.getStride  ((conf.getStrideSize  () == 1) ? 0 : i));
            padding.set (i, (conf.getPaddingSize () == 0) ? 0 : conf.getPadding ((conf.getPaddingSize () == 1) ? 0 : i));
        }

		Shape columnShape = ((Conv) operator.getPeer().getKernel()).getLocalVariableColumn().get()[0].getShape().copy();
		Variable _column = new Variable ("column", columnShape, false);
		_column.initialise (new InitialiserConf().setValue(0));
		columns = new LocalVariable(_column);

		imageShape = ((Conv) operator.getPeer().getKernel()).getImageShape().copy();
		
		Variable biasmultiplier = null;
		if (conf.hasBias()) {
			/*
			 * When multiplied with "bias", the resulting variable should have the same
			 * dimensionality as top and top_diff.
			 */
			biasmultiplier = new Variable ("bias-multiplier", new Shape (new int [] { inputShape[0].countElements(axis + 1) }), false);
			biasmultiplier.initialise (new InitialiserConf().setValue(1));
			
			_biasmultiplier = new LocalVariable(biasmultiplier);

			log.debug(String.format("Local variable %s", biasmultiplier.getName()));
		}
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? Yes, `column` and `biasmultiplier` */
		memoryRequirements.setLocalCPUMemoryRequirements (_column.capacity());
		if (conf.hasBias())
			memoryRequirements.incLocalCPUMemoryRequirements (biasmultiplier.capacity());
		
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
		
		/* Set GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "bias", conf.hasBias() ? 1 : 0);
	}
	
	public void compute (Operator[] previous, Batch batch, Model model, ITask api) {

		/* We want to compute two things; (1) bottomdiff (2) gradients (weightdiff and biasdiff) */

		/* To do this, we need 1. top_diff
		                       2. weight+bias variables from conv
		                       3. the input of conv */

		IDataBuffer weightGradientBuffer, biasGradientBuffer = null;

		IDataBuffer inputDataBuffer, peerInputBuffer, outputDataBuffer;
		int inputStartP, 
			// inputEndP, 
			peerInputStartP;
			// peerInputEndP;

		int axis, filters, channels, batchsize;

		int inputSize, outputSize;

		/* Get input buffer */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();
//		inputEndP = getEndPointer ();

        /* Prepare buffers for gradients: weightGradientBuffer & biasGradientBuffer */
        ModelGradient gradient = batch.getModelGradient(model);
        VariableGradient weightsGradient = gradient.getVariableGradient (operator.getPeer().getId(), 1);
		weightGradientBuffer = weightsGradient.getDataBuffer();
		
		/* Reset gradients */
		weightGradientBuffer.bzero();
		
		VariableGradient biasGradient = null;
        if (conf.hasBias()) {
        	biasGradient = gradient.getVariableGradient (operator.getPeer().getId(), 2);
        	biasGradientBuffer = biasGradient.getDataBuffer();
        	
        	/* Reset gradients */
        	biasGradientBuffer.bzero();
        }
        
        /* Get weight + bias variables from conv */
		int convOpId = operator.getPeer().getId();

		Variable weights = model.getVariable(convOpId, 1);
//        Variable bias = null;
//        if (conf.hasBias()) {
//            bias = model.getVariable(convOpId, 2);
//        }

		/* Get the input of conv operator */
        peerInputBuffer = getPeerInput (batch, api);
		peerInputStartP = getStartPointer ();
		// peerInputEndP = getEndPointer ();

        Variable [] input  = theInput.get(); 
        Variable [] output = theOutput.get();
        
        /* Get output buffer */
		outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		outputDataBuffer.bzero();

		axis      = conf.getAxis();
		filters   = conf.numberOfOutputs();
		channels  = outputShape.get(axis);   // Number of channels of bottom_diff
		batchsize = outputShape.countElements(0, axis);
		// groups    = conf.numberOfGroups();

        inputSize   =  input[0].getShape() .countElements(axis) * input[0].getType().sizeOf();  //Size of each top_diff
        outputSize  =  output[0].getShape().countElements(axis) * output[0].getType().sizeOf(); //Size of bottom_diff

		/* Prepare buffer for column (This will be re-used throughout) */
        IDataBuffer columnbuffer = columns.get()[0].getDataBuffer();

        for (int n = 0; n < batchsize; ++n) { 

            int topdiff_offset = n * inputSize;
            int bottom_offset  = n * outputSize;

            /* Weight */

            int M = filters ;
            int N = input[0].getShape().countElements(axis + 1);
            int K = weights.getShape().countElements(1);

            weight_cpu_gemm (peerInputBuffer,  (peerInputStartP + bottom_offset),
                             columnbuffer, channels,
                             inputDataBuffer, (inputStartP + topdiff_offset),
                             weightGradientBuffer,
                             M, N, K);

            if (conf.hasBias()) {
                bias_cpu_gemv(biasGradientBuffer, inputDataBuffer, (inputStartP + topdiff_offset), M, N);
            }

            /* (2) gradient w.r.t. bottom data, if necessary */

			if(! api.isMostUpstream(operator.getPeer())){

				model.readLock();

				IDataBuffer weightsbuffer = weights.getDataBuffer();

				backward_cpu_gemm(outputDataBuffer, bottom_offset,
						columnbuffer, channels,
						inputDataBuffer, (inputStartP + topdiff_offset),
						weightsbuffer, imageShape,
						M, N, K);

				model.readUnlock();
			}
        }
        
        /* Store output in batch for downstream operators */
		batch.setOutput(operator.getId(), outputDataBuffer);
		
        log.debug("ConvGradient operator done...");
	}

	private void weight_cpu_gemm (IDataBuffer bottom,  int bottom_offset,
                                  IDataBuffer columnbuffer, int channels,
                                  IDataBuffer topdiff, int topdiff_offset,
                                  IDataBuffer weightdiff,
                                  int M, int N, int K) {

		/* Note that we will accumulate diffs. So we use beta as 1 */

        int lda = N;
        int ldb = N;
        int ldc = K;

        int Alimit = M * N * 4;
        int Blimit = K * N * 4;
        int Climit = M * K * 4;

        if( ((Conv) operator.getPeer().getKernel()).getScalar() ){ /* 1x1 case */

            /* Use bottom directly */

            BLAS.getInstance().sgemm("N", "T",
                    M, K, N,
                    1F,
                    topdiff, topdiff_offset, Alimit + topdiff_offset, lda,
                    bottom,   bottom_offset, Blimit + bottom_offset,  ldb,
                    1F, /* beta */
                    weightdiff,           0, Climit, ldc);

        } else {

			/* Clean column buffer first (I know that this will eventually be over-written ..) */
			IDataBufferIterator it = columnbuffer.getIterator();
			while (it.hasNext()) {
				int offset = it.next();
				columnbuffer.putFloat (offset, 0);
			}

            /* We need to derive column buffer from bottom first */
            imageToColumn(bottom, bottom_offset, columnbuffer, channels);

            BLAS.getInstance().sgemm("N", "T",
                    M, K, N,
                    1F,
                    topdiff,      topdiff_offset, Alimit + topdiff_offset, lda,
                    columnbuffer,              0, Blimit                 , ldb,
                    1F,
                    weightdiff,                0, Climit, ldc);
        }
	}

    private void backward_cpu_gemm (IDataBuffer bottomdiff,  int bottomdiff_offset,
                                   IDataBuffer columnbuffer, int channels,
                                   IDataBuffer topdiff, int topdiff_offset,
                                   IDataBuffer weightbuffer, Shape imageShape,
                                   int M, int N, int K){
        
        int lda = K;
        int ldb = N;
        int ldc = N;

        int Alimit = M * K * 4;
        int Blimit = M * N * 4;
        int Climit = K * N * 4;

        if( ((Conv) operator.getPeer().getKernel()).getScalar() ){ /* 1x1 case */

            BLAS.getInstance().sgemm ("T", "N",
                    K, N, M,
                    1F,
                    weightbuffer  ,                    0, Alimit,                     lda,
                    topdiff ,       topdiff_offset, Blimit + topdiff_offset,    ldb,
                    0F,
                    bottomdiff , bottomdiff_offset, Climit + bottomdiff_offset, ldc);

        }else{

            BLAS.getInstance().sgemm ("T", "N",
                    K, N, M,
                    1F,
                    weightbuffer  ,               0, Alimit,                  lda,
                    topdiff ,  topdiff_offset, Blimit + topdiff_offset, ldb,
                    0F,
                    columnbuffer ,          0, Climit,                  ldc); /* Column buffer will be over written with new values */

            columnToImage2D (bottomdiff, bottomdiff_offset, columnbuffer, channels,
                    imageShape.get(1),   imageShape.get(2), /* Image  height & width */
                    kernel.get(0),  kernel.get(1),          /* Kernel height & width */
                    padding.get(0), padding.get(1),         /* ...                   */
                    stride.get(0),  stride.get(1));

        }
    }

	private void bias_cpu_gemv (IDataBuffer biasdiff, IDataBuffer topdiff, int topdiff_offset,
                                int filters, int columns) {

		IDataBuffer biasmultiplier = _biasmultiplier.get()[0].getDataBuffer();

        int M = filters; // number of filters (num_output_)
        int N = columns; // size of a row in top_diff (i.e. out_spatial_dim_ ; Each row contains all outputs from a filter)
        int K = 1;
        int lda = N;
        int ldb = K;
        int ldc = K;

        int Alimit = M * N * 4; /* top_diff */
        int Blimit = N * K * 4; /* bias_multiplier */
        int Climit = M * K * 4; /* bias_diff */

        /* TODO Check correctness */
        BLAS.getInstance().sgemm ("N", "N",
                M, K, N,
                1F,
                topdiff ,        topdiff_offset, Alimit + topdiff_offset, lda,
                biasmultiplier ,              0, Blimit                 , ldb,
                1F,
                biasdiff ,                    0, Climit                 , ldc);

	}

    private void imageToColumn (IDataBuffer bottom, int offset, IDataBuffer columnbuffer, int channels) {

        if (spatialDimensions != 2)
            throw new UnsupportedOperationException("error: multi-dimensional convolution is not yet supported");

        imageToColumn2D(bottom, offset, columnbuffer, channels,

                imageShape.get(1), imageShape.get(2), /* Image  height & width */
                kernel.get(0), kernel.get(1), /* Kernel height & width */
                padding.get(0), padding.get(1), /* ...                   */
                stride.get(0), stride.get(1)
        );
    }

	private void imageToColumn2D (IDataBuffer bottom, int offset, IDataBuffer column, int channels,

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
						value = bottom.getFloat(offset + (((c_ * imageHeight + h_) * imageWidth + w_) * DataType.FLOAT.sizeOf()));
					}

					column.putFloat(index, value);
				}
			}
		}
	}

	private void columnToImage2D (IDataBuffer image, int offset, IDataBuffer column, int channels,

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

					int index_in_column = ((columnelement * output_h + h) * output_w + w) * DataType.FLOAT.sizeOf();
					float value_to_add  = column.getFloat(index_in_column);

					if (h_ >= 0 && w_ >= 0 && h_ < imageHeight && w_ < imageWidth) {

						int index_in_image = offset + (((c_ * imageHeight + h_) * imageWidth + w_) * DataType.FLOAT.sizeOf()) ;
						float current_value = image.getFloat(index_in_image);

						image.putFloat(index_in_image, current_value + value_to_add);
					}
				}
			}
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
