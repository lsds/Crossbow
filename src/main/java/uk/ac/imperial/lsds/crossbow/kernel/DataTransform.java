package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.kernel.conf.DataTransformConf;
import uk.ac.imperial.lsds.crossbow.model.LocalVariable;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.model.Variable;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ThreadLocalRandom;

public class DataTransform extends Kernel {
	
	private final static Logger log = LogManager.getLogger(DataTransform.class);

	private DataTransformConf conf;
	
	private LocalVariable _meanvalues = null;
	
	public DataTransform (DataTransformConf conf) {
		this.conf = conf;
	}
	
	@Override
	public IKernel setup (Shape [] inputShape, Model model) {
		
		log.debug(String.format("Setup kernel for operator %s", operator.getName()));
		
		if (inputShape.length > 1)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable input = new Variable ("input", inputShape [0], true);
		theInput = new LocalVariable(input);
		
		log.debug(String.format("Input variable %s", input.getName()));
		
		int channels = inputShape [0].numberOfChannels();
		int   height = inputShape [0].height();
		int    width = inputShape [0].width ();
		
		/* Check whether crop size is valid */
		if((conf.getCropSize() > height) || (conf.getCropSize() > width)) {
			
			throw new IllegalArgumentException (String.format("error: cannot crop %d x %d image to %d x %d", 
					height, width, conf.getCropSize(), conf.getCropSize()));
		}
		
		/* Configure the output shape */
		
		if (conf.getCropSize() > 0)
			outputShape = new Shape (new int [] {inputShape [0].numberOfExamples(), channels, conf.getCropSize(), conf.getCropSize() });
		else
			outputShape = inputShape [0].copy();
		
		Variable output = new Variable ("output", outputShape, true);
		theOutput = new LocalVariable (output);
		
		log.debug(String.format("Output variable %s", output.getName()));
		
		/* Local variable(s) */
		
		Variable var = null;
		
		if (conf.subtractMean ()) {
			
			if (conf.hasMeanImage ()) {
			
				var = new Variable ("means", new Shape (new int [] { channels, height, width }), false);
			
				/* Load mean image file into a byte buffer and copy it to the local variable */
				loadFile (conf.getMeanImageFilename (), var.getDataBuffer());
			
				_meanvalues = new LocalVariable (var);
			}
			else {
			
				/* Store (x, y, z) into local variable */
			
				var = new Variable ("means", new Shape (new int [] { channels }), false);
			
				float [] pixels = conf.getMeanPixelValues ();
			
				var.getDataBuffer().putFloat(0, pixels [0]);
				var.getDataBuffer().putFloat(0, pixels [1]);
				var.getDataBuffer().putFloat(0, pixels [2]);
			
				_meanvalues = new LocalVariable (var);
			}
			
			log.debug(String.format("Local variable %s", var.getName ()));
		}
		
		/* 
		 * Set memory requirements 
		 */
		
		/* Set output, by default */
		memoryRequirements.setOutputMemoryRequirements(output.capacity());
		
		/* Are there any model variables? No */
		memoryRequirements.setModelMemoryRequirements (0);
		
		/* Are there any CPU-specific local variables? Yes, `means` */
		if (conf.subtractMean())
			memoryRequirements.setLocalCPUMemoryRequirements (var.capacity());
		
		/* Are there any GPU-specific local variables? Yes, `means` and `randoms` */
		if (conf.subtractMean())
			memoryRequirements.incLocalGPUMemoryRequirements (var.capacity());
		
		if (conf.getCropSize() > 0)
			memoryRequirements.incLocalGPUMemoryRequirements (12 * inputShape[0].numberOfExamples());
		
		return this;
	}
	
	@Override
	public void GPURegister () {

		log.debug(String.format ("Register kernel with GPU for operator %s", operator.getName()));

		int id = operator.getId();
		String name = this.getClass().getSimpleName();
		
		int numberoflocalvariables = 0;
		
		if (conf.getCropSize() > 0 || conf.getMirror())
			numberoflocalvariables ++;
		
		if (conf.subtractMean())
			numberoflocalvariables ++;
		
		/* 1 input, variable local variables, 1 output */
		TheGPU.getInstance().setKernel (id, name, 1, numberoflocalvariables, 1, (isLossKernel() || isAccuracyKernel()));
		
		Variable []  input =  theInput.getInitialValue();
		Variable [] output = theOutput.getInitialValue();
		
		int examples = input[0].getShape().numberOfExamples();
		
		/* Set input */
		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
		
		/* Set output */
		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
		
		/* Set local variables */
		int localvariableid = 0;
		
		if (conf.subtractMean ()) {
			Variable [] local = _meanvalues.getInitialValue ();
			TheGPU.getInstance().setKernelLocalVariable (id, localvariableid ++, "means", local[0].getShape().array(), local[0].capacity(), true);
			/* Initialise `_meanvalues` variable on GPU */
			TheGPU.getInstance().setKernelLocalVariableData (id, 0, local[0].getDataBuffer());
		}
		
		if (conf.getCropSize() > 0 || conf.getMirror()) {
			/* The second local variable is GPU-specific used to store generated random numbers on device memory */
			TheGPU.getInstance().setKernelLocalVariable (id, localvariableid ++, "randoms", new int [] { 3 * examples }, (12 * examples), false);
		}
		
		/* Configure GPU kernel-specific parameters */
		TheGPU.getInstance().setKernelConfigurationParameters (id, 5);
		
		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 0,     "cropSize", conf.getCropSize    ());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 1,       "mirror", conf.getMirror      () ? 1 : 0);
		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 2,  "scaleFactor", conf.getScaleFactor ());
		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 3, "subtractMean", conf.subtractMean   () ? 1 : 0);
		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 4, "hasMeanImage", conf.hasMeanImage   () ? 1 : 0);
		
		return;
	}
	
	@Override
	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {

		log.debug(String.format("Compute kernel for operator %s", operator.getName()));
		
		/* Assert that this operator is the most upstream */
		if (previous != null)
			throw new IllegalArgumentException (String.format("error: invalid number of inputs for operator %s", operator.getName()));
		
		Variable []  input =  theInput.get();
		Variable [] output = theOutput.get();
		
		int examples = input[0].getShape().numberOfExamples ();
		int channels = input[0].getShape().numberOfChannels ();
		
		int inputImageHeight = input[0].getShape().height ();
		int inputImageWidth  = input[0].getShape().width ();
		
		/* Input image n starts at `n x imageOffset` */
		int inputImageOffset = channels * inputImageHeight * inputImageWidth * input[0].getType().sizeOf();
		
		int outputImageHeight = output[0].getShape().height ();
		int outputImageWidth  = output[0].getShape().width ();
		
		/* Output image n starts at `n x croppedImageOffset` */
		int outputImageOffset = channels * outputImageHeight * outputImageWidth * output[0].getType().sizeOf();
		
		/* Get input buffer */
		IDataBuffer inputDataBuffer = getCurrentInput (batch, api);
		int inputStartP = getStartPointer ();
		
		/* Get output buffer */
		IDataBuffer outputDataBuffer = getCurrentOutput (batch, api);
		output[0].wrap (outputDataBuffer);
		
		/* Get configuration variables */
		int cropSize = conf.getCropSize ();
		float scaleFactor = conf.getScaleFactor ();
		
		/* Get local variable "means" */
		Variable [] meanimage = null; 
		IDataBuffer means = null;
		
		if (conf.subtractMean ()) {
			meanimage = _meanvalues.get();
			means = meanimage[0].getDataBuffer();
		}
		
		boolean mirror = conf.getMirror () && (ThreadLocalRandom.current().nextInt(2) == 1);
		boolean training = (! api.isValidationTask());
		
		for (int n = 0; n < examples; ++n) {
			
			/* Find current position in input/output buffers */
			int  inputImagePos = (n *  inputImageOffset) + inputStartP;
			int outputImagePos = (n * outputImageOffset);
			
			int heightOffset = 0, widthOffset = 0;
			
			if (cropSize > 0) {
				if (training) {
					heightOffset = ThreadLocalRandom.current().nextInt (inputImageHeight - cropSize + 1);
					 widthOffset = ThreadLocalRandom.current().nextInt (inputImageWidth  - cropSize + 1);
				} else {
					heightOffset = (inputImageHeight - cropSize) / 2;
					 widthOffset = (inputImageWidth  - cropSize) / 2;
				}
			}
			
			for (int c = 0 ; c < channels; ++c) {
				for (int h = 0 ; h < outputImageHeight; ++h) {
					for (int w = 0 ; w < outputImageWidth; ++w) {
						
						int  inputIndex = (c * inputImageHeight  + heightOffset + h) * inputImageWidth + widthOffset + w;
						int outputIndex = (c * outputImageHeight + h) * outputImageWidth + ((mirror) ? (outputImageWidth - 1 - w) : w);
						
						int  inputPixelPos =  inputImagePos + ( inputIndex *  input[0].getType().sizeOf ());
						int outputPixelPos = outputImagePos + (outputIndex * output[0].getType().sizeOf ());
						
						float pixel = inputDataBuffer.getFloat(inputPixelPos);
						
						/* Subtract mean */
						if (conf.subtractMean ()) {
							
							if (conf.hasMeanImage ())
								pixel -= means.getFloat(meanimage[0].getType().sizeOf() * inputIndex);
							else
								pixel -= means.getFloat(meanimage[0].getType().sizeOf() * c);
						}
						
						/* Scale */
						if (scaleFactor != 1)
							pixel *= scaleFactor;
						
						/* Write pixel to output */
						outputDataBuffer.putFloat(outputPixelPos, pixel);
					}
				}
			}
		}
		
		/* Store output in batch for downstream operators */
		batch.setOutput(operator.getId(), outputDataBuffer);
	}
	
	@SuppressWarnings("resource")
	private void loadFile (String filename, IDataBuffer image) {
		File file = new File(filename);
		FileChannel channel;
		MappedByteBuffer buffer = null;
		try {
			
			channel = new RandomAccessFile(file, "rw").getChannel();
			buffer = channel.map(FileChannel.MapMode.READ_WRITE, 0, channel.size()).load();
			buffer.order(ByteOrder.LITTLE_ENDIAN);
		
			/* Fill local variable buffer */
			IDataBufferIterator iterator = image.getIterator();
			while (iterator.hasNext()) {
				int offset = iterator.next();
				image.putFloat(offset, buffer.getFloat(offset));
			}
		
			channel.close();
		
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		return;
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
		return true;
	}
	
	public boolean allowsOutputOverwrite () {
		return true;
	}
	
	public boolean allowsInputOverwrite () {
		return false;
	}
}
