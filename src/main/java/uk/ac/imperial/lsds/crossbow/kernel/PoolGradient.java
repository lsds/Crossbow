package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;

import java.nio.BufferOverflowException;

public class PoolGradient extends Kernel {
	
	private final static Logger log = LogManager.getLogger (PoolGradient.class);
	
	PoolConf conf;

    int __pooledHeight, __pooledWidth;

    int  kernelWidth,  kernelHeight;
    int paddingWidth, paddingHeight;
    int  strideWidth,  strideHeight;

    int examples, channels, height, width;

    public PoolGradient (PoolConf conf) {
		this.conf = conf;
	}

	public PoolGradient setup (Shape[] inputShape, Model model) {

		log.debug(String.format("Setup kernel for operator %s", operator.getName()));

        kernelWidth  =  kernelHeight = 0;
        paddingWidth = paddingHeight = 0;
        strideWidth  =  strideHeight = 0;
		
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


        /*
		 * Number of examples, number of channels, height and width
		 * are the 1st, 2nd, 3rd, and 4th dimension respectively.
		 */
        examples = outputShape.numberOfExamples();
        channels = outputShape.numberOfChannels();
        height   = outputShape.height();
        width    = outputShape.width();

        if (conf.isGlobal()) {
			/* Kernel size (incl. height and width) must not be set */
            if ((conf.getKernelSize() != 0) || (conf.getKernelWidth() != 0) || (conf.getKernelHeight() != 0)) {
                System.err.println("error: cannot override global kernel size configuration");
                System.exit(1);
            }

            kernelHeight = height;
            kernelWidth  =  width;

			/* Override padding and stride configuration */
            paddingHeight = paddingWidth = 0;
            strideHeight  =  strideWidth = 1;

        } else {
			/* Either the kernel size is set, or the kernel height and width. If
			 * the kernel size is not set, both height and width must be set.
			 *
			 * Same conditions apply for padding and stride size.
			 */
            if (conf.getKernelSize() == 0) {

                if ((conf.getKernelHeight() == 0) || (conf.getKernelWidth() == 0)) {
                    System.err.println("error: both kernel height and width are required");
                    System.exit(1);
                }

                kernelHeight = conf.getKernelHeight();
                kernelWidth  = conf.getKernelWidth();

            } else {
                kernelHeight = kernelWidth = conf.getKernelSize();
            }

            if (conf.getPaddingSize() < 0) {

                if ((conf.getPaddingHeight() < 0) || (conf.getPaddingWidth() < 0)) {
                    System.err.println("error: both padding height and width are required");
                    System.exit(1);

                    paddingHeight = conf.getPaddingHeight();
                    paddingWidth  = conf.getPaddingWidth();
                }

            } else {
                paddingHeight = paddingWidth = conf.getPaddingSize();
            }

            if (conf.getStrideSize() == 0) {

                if ((conf.getStrideHeight() == 0) || (conf.getStrideWidth() == 0)) {
                    System.err.println("error: both stride height and width are required");
                    System.exit(1);
                }

                strideHeight = conf.getStrideHeight();
                strideWidth  = conf.getStrideWidth();

            } else {
                strideHeight = strideWidth = conf.getStrideSize();
            }
        }

        if (paddingHeight != 0 || paddingWidth != 0) {

            if (conf.getMethod().equals(PoolMethod.STOCHASTIC)) {
                System.err.println("error: stochastic pooling does not support padding");
                System.exit(1);
            }

            if (paddingHeight >= kernelHeight) {
                System.err.println("error: padding height must be less than kernel height");
                System.exit(1);
            }

            if (paddingWidth >= kernelWidth) {
                System.err.println("error: padding width must be less than kernel width");
                System.exit(1);
            }
        }

        log.debug(String.format("Heights: image %d padding %d kernel %d stride %d", height, paddingHeight, kernelHeight, strideHeight));
        log.debug(String.format("Widths : image %d padding %d kernel %d stride %d", width, paddingWidth, kernelWidth, strideWidth));

        __pooledHeight = (int) (Math.ceil((double) (height + 2 * paddingHeight - kernelHeight) / strideHeight)) + 1;
        __pooledWidth  = (int) (Math.ceil((double) (width  + 2 * paddingWidth  - kernelWidth ) / strideWidth )) + 1;

        log.debug(String.format("Pool height is %d, width is %d", __pooledHeight, __pooledWidth));

        if (paddingHeight != 0 || paddingWidth != 0) {

            if (((__pooledHeight - 1) * strideHeight) >= (height + paddingHeight))
                --__pooledHeight;

            if (((__pooledWidth  - 1) * strideWidth ) >= (width  + paddingWidth ))
                --__pooledWidth;

			/* Check bounds */

            if (((__pooledHeight - 1) * strideHeight) >= (height + paddingHeight)) {
                System.err.println("error: invalid pool height");
                System.exit(1);
            }

            if (((__pooledWidth  - 1) *  strideWidth) >= (width  +  paddingWidth)) {
                System.err.println("error: invalid pool width");
                System.exit(1);
            }
        }
        
        /* 
		 * Set memory requirements 
		 */
		
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
		
		/* 1 input, 0 local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0, input[0].getShape().array(), input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id, output[0].getShape().array(), output[0].capacity());
		
		return;
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));

        int hstart, hend;
        int wstart, wend;
        int pool_size;

        int   buttomdiffBufferIndex, topdiffBufferIndex;
        float bottomdiffValue, topdiffValue;

        IDataBuffer outputDataBuffer, inputDataBuffer, poolIndexBuffer; // , peerInputBuffer;
        int inputStartP, peerInputStartP;

        float topdiff_val ;
        
        int index_offset, topdiff_offset;

        Variable [] input, output;

		/* Get thread-local variables */
        input  =  theInput.get(); 
        output = theOutput.get();
        
        /* Get input buffer */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();
        
        /* Get output buffer from pool */
		outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
        outputDataBuffer.bzero();

        /* Get the start of the input of pooling operator i.e. bottom */
//        peerInputBuffer = getPeerInput (batch, api);
		peerInputStartP = getStartPointer ();

        /*
         * TODO
         * 
         * Enable iterator even for circular data buffer.
         * This will remove the need to keep counter and
         * topdiff_offset in the code below:
         */

        int __input_offset, __output_offset;
        int pooled_index, bottomdiff_index;

		switch (conf.getMethod()) {
		
		case MAX:
			
			/* Get localVar (containing indices) from Pool operator */
	        LocalVariable localVar  = ((Pool) operator.getPeer().getKernel()).getLocalVariable();
	        Variable [] indices     = localVar.get();
	        poolIndexBuffer         = indices[0].getDataBuffer();
	        
            int counter = 0 ;
            IDataBufferIterator indexIterator = poolIndexBuffer.getIterator();
            topdiff_offset = input[0].getType().sizeOf() ;
            
            while (indexIterator.hasNext()) {

                index_offset   = indexIterator.next();
                /* 'poolIndexBuffer' contains indices based on the underlying input buffer (bottom_data) of Pooling op */
                bottomdiff_index = poolIndexBuffer.getInt(index_offset) - peerInputStartP;
                topdiff_val  = inputDataBuffer.getFloat(inputStartP + (topdiff_offset * counter) );

                outputDataBuffer.putFloat(bottomdiff_index, topdiff_val);
                counter++;
            }

			break;
		
		case AVERAGE:

            __input_offset = __output_offset = 0;

            for (int n = 0; n < examples; ++n) {
                for (int c = 0; c < channels; ++c) {

                    for (int ph = 0; ph < __pooledHeight; ++ph) {
                        for (int pw = 0; pw < __pooledWidth; ++pw) {

                            hstart = ph * strideHeight - paddingHeight;
                            wstart = pw * strideWidth  - paddingWidth;

                            hend = Math.min (hstart + kernelHeight, height + paddingHeight);
                            wend = Math.min (wstart + kernelWidth,  width  + paddingWidth);

                            pool_size = (hend - hstart) * (wend - wstart);

                            hstart = Math.max (hstart, 0);
                            wstart = Math.max (wstart, 0);

                            hend = Math.min (hend, height);
                            wend = Math.min (wend, width);

                            /* pooled_index is an index based on top-diff (input) */
                            pooled_index = ph * __pooledWidth + pw;
                            /* convert the pooled_index to an index based on the actual input buffer (top-diff buffer) */
                            topdiffBufferIndex = inputStartP + (__input_offset + pooled_index) * input[0].getType().sizeOf();

                            /* To fill the bottom_diff (i.e. the output) */
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {

                                    /* bottomdiff_index is an index based on the logical bottom_diff (output) */
                                    bottomdiff_index = h * width + w ;
                                    /* convert the bottomdiff_index to an index based on the actual output buffer */
                                    buttomdiffBufferIndex = ((__output_offset + bottomdiff_index) * output[0].getType().sizeOf());

                                    if (buttomdiffBufferIndex >= outputDataBuffer.limit())
                                        throw new BufferOverflowException();

                                    bottomdiffValue = outputDataBuffer.getFloat(buttomdiffBufferIndex);
                                    topdiffValue    = inputDataBuffer.getFloat(topdiffBufferIndex);

                                    /* Accumulating */
                                    outputDataBuffer.putFloat (buttomdiffBufferIndex, bottomdiffValue + (topdiffValue / pool_size));

                                    if (buttomdiffBufferIndex < 0) {
                                        System.err.println(String.format("Oops: input index is %d, start pointer is %d", buttomdiffBufferIndex, 0));
                                        System.exit(1);
                                    }
                                }
                            }
                        }
                    }

					/* Increment offsets */
                    __input_offset  +=  input[0].getShape().offset(0, 1);
                    __output_offset += output[0].getShape().offset(0, 1);
                }
            }
            
            break;
		
		case STOCHASTIC:
			throw new UnsupportedOperationException("error: stochastic pooling method is not yet implemented");
		
		default:
			throw new IllegalArgumentException("error: invalid pooling method");
		}

        /* Store output in batch for downstream operators */
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
