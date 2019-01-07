package uk.ac.imperial.lsds.crossbow.kernel;

import java.nio.BufferOverflowException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.CudnnKernelType;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;

public class Pool extends Kernel {
	
	private final static Logger log = LogManager.getLogger (Pool.class);
	
	PoolConf conf;
	
	LocalVariable _local;
	
	int __pooledHeight, __pooledWidth;
	
	int  kernelWidth,  kernelHeight;
	int paddingWidth, paddingHeight;
	int  strideWidth,  strideHeight;
	
	int examples, channels, height, width; 
	
	public Pool (PoolConf conf) {
		this.conf = conf;
	}
	
	public Pool setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		kernelWidth  =  kernelHeight = 0;
		paddingWidth = paddingHeight = 0;
		strideWidth  =  strideHeight = 0;
		
		if (inputShape[0].dimensions() != 4) {
			System.err.println(String.format("error: invalid input shape (expected a 4-D shape, found a %d-D one)", inputShape[0].dimensions()));
			System.exit(1);
		}
		
		Variable input = new Variable ("input", inputShape[0], true);
		theInput = new LocalVariable (input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		/* 
		 * Number of examples, number of channels, height and width
		 * are the 1st, 2nd, 3rd, and 4th dimension respectively.
		 */
		examples = inputShape[0].numberOfExamples();
		channels = inputShape[0].numberOfChannels();
		height   = inputShape[0].height();
		width    = inputShape[0].width();
		
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
		
		/* Configure the output shape */
		
		outputShape = new Shape (new int [] { examples, channels, __pooledHeight, __pooledWidth });
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));
		
		/* Configure local variables */
		
		Variable v;
		
		switch (conf.getMethod()) {
			case MAX: 
				v = new Variable ("indices", new Shape (new int [] { examples, channels, __pooledHeight, __pooledWidth }), false, DataType.INT); 
				break;
			case STOCHASTIC: 
				v = new Variable ("indices", new Shape (new int [] { examples, channels, __pooledHeight, __pooledWidth }), false, DataType.INT);
				break;
			case AVERAGE:
				/* Fallback to default */
			default:
				v = null;
		}
		
		if (v != null) {
			v.initialise (new InitialiserConf().setValue(0));
			_local = new LocalVariable(v);
			
			log.debug(String.format("Local variable %s", v.getName()));
		}
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? Yes, `indices` */
		if (v != null)
			memoryRequirements.setLocalCPUMemoryRequirements (v.capacity());
		
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
		
		/* Set cuDNN kernel */
		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.POOL.getId());
		
		int [] dimensions = new int [4];
		
		input[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		output[0].getShape().getNCHW (dimensions);
		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
		
		TheGPU.getInstance().cudnnSetPoolingMode (id, conf.getMethod().getId());
		
		TheGPU.getInstance().cudnnSetPoolingDescriptor (id, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth);
		
		return;
	}
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
		
		log.debug(String.format("Compute kernel for operator %s", operator.getName()));

		if (previous != null && previous.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		int hstart, hend;
		int wstart, wend;
        int pool_size;
		
		int   inputBufferIndex, outputBufferIndex;
		float inputValue, outputValue;
		
		IDataBuffer inputDataBuffer, outputDataBuffer, poolIndexBuffer; /* input, output and local buffers */
		int inputStartP, inputEndP;
		
		Variable [] input, output, indices;
		
		/* Get thread-local variables */
		input  =  theInput.get();
		output = theOutput.get();
		
		/* Get input buffer */
		inputDataBuffer = getCurrentInput (batch, api);
		inputStartP = getStartPointer ();
		inputEndP = getEndPointer ();
		
		/* Get output buffer */
		outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap(outputDataBuffer);
		
		
		IDataBufferIterator iterator;
		int offset;
		
		int __input_offset, __output_offset;
		int pooled_index, bottom_index;
        boolean isFirst;

		switch (conf.getMethod()) {
		
		case MAX:
			
			indices = _local.get();
			poolIndexBuffer = indices[0].getDataBuffer();
			
			/* Initialise _local variable to -1 */
			iterator = poolIndexBuffer.getIterator();
			while (iterator.hasNext()) {
				offset = iterator.next();
				poolIndexBuffer.putInt(offset, -1);
			}
			
			/* Initialise output to Float.MIN_VALUE */
			iterator = outputDataBuffer.getIterator();
			while (iterator.hasNext()) {
				offset = iterator.next();
				outputDataBuffer.putFloat(offset, -Float.MAX_VALUE);
			}
			
			__input_offset = __output_offset = 0;

			for (int n = 0; n < examples; ++n) {
				for (int c = 0; c < channels; ++c) {
					
					for (int ph = 0; ph < __pooledHeight; ++ph) {
						for (int pw = 0; pw < __pooledWidth; ++pw) {
							
							hstart = ph * strideHeight - paddingHeight;
							wstart = pw * strideWidth  - paddingWidth;
							
							hend = Math.min (hstart + kernelHeight, height);
							wend = Math.min (wstart + kernelWidth,  width);
							
							hstart = Math.max (hstart, 0);
							wstart = Math.max (wstart, 0);

                            /* 'pooled_index' is an index based on the logical top_data */
                            pooled_index = ph * __pooledWidth + pw;
                            /* convert the pooled_index to an index based on the underlying output-buffer */
							outputBufferIndex = (__output_offset + pooled_index) * output[0].getType().sizeOf();
							
							isFirst = true;
							
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {

                                    /* 'bottom_index' is an index based on the logical bottom_data */
                                    bottom_index = h * width + w ;
                                    /* convert the bottom_index to an index based on the underlying input-buffer */
									inputBufferIndex = inputStartP + ((__input_offset + bottom_index) * input[0].getType().sizeOf());
									
									if (inputBufferIndex >= inputEndP)
										throw new BufferOverflowException();
									
									inputValue  = inputDataBuffer.getFloat(inputBufferIndex);
									outputValue = outputDataBuffer.getFloat(outputBufferIndex);
									
									if (isFirst) {
										isFirst = false;
                                        /* Using the same index, the top-buffer stores the pooled value
                                        *   while the pool-index-buffer stores the index where the value is picked from */
										outputDataBuffer.putFloat (outputBufferIndex, inputValue);
										poolIndexBuffer.putInt (outputBufferIndex, inputBufferIndex); // - start);
										
									} else {
										if (inputValue > outputValue){
											
											outputDataBuffer.putFloat (outputBufferIndex,  inputValue);
											poolIndexBuffer.putInt (outputBufferIndex,  inputBufferIndex); // - start);

                                            if (inputBufferIndex - inputStartP < 0) {
												System.err.println(String.format("Oops: input index is %d, start pointer is %d", inputBufferIndex, inputStartP));
												System.exit(1);
											}
										}
										
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
		
		case AVERAGE:

			/* Initialise output buffer to 0 */
            iterator = outputDataBuffer.getIterator();
            while (iterator.hasNext()) {
                offset = iterator.next();
                outputDataBuffer.putFloat(offset, 0);
            }

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

                            /* pooled_index is an index based on the logical top_data (output) */
                            pooled_index = ph * __pooledWidth + pw;
                            /* convert the pooled_index to an index based on the underlying output-buffer */
                            outputBufferIndex = (__output_offset + pooled_index) * output[0].getType().sizeOf();

                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {

                                    /* bottom_index is an index based on the logical bottom_data (input) */
                                    bottom_index = h * width + w ;
                                    /* convert the bottom_index to an index based on the underlying input-buffer */
                                    inputBufferIndex = inputStartP + ((__input_offset + bottom_index) * input[0].getType().sizeOf());

                                    if (inputBufferIndex >= inputEndP)
                                        throw new BufferOverflowException();

                                    inputValue  = inputDataBuffer.getFloat(inputBufferIndex);
                                    outputValue = outputDataBuffer.getFloat(outputBufferIndex);
                                    /* Accumulating */
                                    outputDataBuffer.putFloat (outputBufferIndex, outputValue + inputValue);

                                    if (inputBufferIndex - inputStartP < 0) {
                                        System.err.println(String.format("Oops: input index is %d, start pointer is %d", inputBufferIndex, inputStartP));
                                        System.exit(1);
                                    }
                                }
                            }

                            /* After accumulating, we compute the mean by dividing with pool-size */
                            outputValue = outputDataBuffer.getFloat(outputBufferIndex);
                            outputDataBuffer.putFloat (outputBufferIndex, outputValue / pool_size);
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
	
		batch.setOutput(operator.getId(), outputDataBuffer);
	}

	public LocalVariable getLocalVariable () {
        return _local;
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
