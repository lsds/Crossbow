package uk.ac.imperial.lsds.crossbow.kernel.conf;

import java.util.Arrays;

import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;

public class ConvConf implements IConf {
	
	private int outputs;
	
	private boolean bias;
	
	private int axis;
	
	private InitialiserConf weightInitialiserConf, biasInitialiserConf;
	
	private int paddingSize;
	private int [] padding;
	
	private int kernelSize;
	int [] kernel;
	
	private int strideSize;
	private int [] stride;
	
	private int groups;
	
	private float weightsLearningRateMultiplier, biasLearningRateMultiplier;
	
	public ConvConf () {
		
		outputs = 1;
		
		bias = true;
		
		axis = 1;
		
		weightInitialiserConf = new InitialiserConf();
		biasInitialiserConf = new InitialiserConf();
		
		paddingSize = 0;
		padding = null;
		
		kernelSize = 0;
		kernel = null;
		
		strideSize = 0;
		stride = null;
		
		groups = 1;
		
		weightsLearningRateMultiplier = biasLearningRateMultiplier = 1;
	}
	
	public int numberOfOutputs () {
		return outputs;
	}
	
	public ConvConf setNumberOfOutputs (int outputs) {
		this.outputs = outputs;
		return this;
	}
	
	public boolean hasBias () {
		return bias;
	}

	public ConvConf setBias (boolean bias) {
		this.bias = bias;
		return this;
	}
	
	public InitialiserConf getWeightInitialiser () {
		return weightInitialiserConf;
	}

	public ConvConf setWeightInitialiser (InitialiserConf weightInitialiserConf) {
		this.weightInitialiserConf = weightInitialiserConf;
		return this;
	}
	
	public InitialiserConf getBiasInitialiser () {
		return biasInitialiserConf;
	}

	public ConvConf setBiasInitialiser (InitialiserConf biasInitialiserConf) {
		this.biasInitialiserConf = biasInitialiserConf;
		return this;
	}

	public int getAxis () {
		return axis;
	}

	public ConvConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
	
	public ConvConf setPaddingSize (int paddingSize) {
		this.paddingSize = paddingSize;
		this.padding = new int [paddingSize];
		Arrays.fill(this.padding, 0);
		return this;
	}
	
	public ConvConf setPaddingHeight (int paddingHeight) {
		this.padding[0] = paddingHeight;
		return this;
	}
	
	public ConvConf setPaddingWidth (int paddingWidth) {
		this.padding[1] = paddingWidth;
		return this;
	}
	
	public ConvConf setPadding (int ndx, int val) {
		this.padding[ndx] = val;
		return this;
	}
	
	public int getPaddingSize () {
		return paddingSize;
	}
	
	public int getPaddingHeight () {
		return padding[0];
	}
	
	public int getPaddingWidth () {
		return padding[1];
	}
	
	public int getPadding (int ndx) {
		return padding[ndx];
	}
	
	public ConvConf setKernelSize (int kernelSize) {
		this.kernelSize = kernelSize;
		this.kernel = new int [kernelSize];
		Arrays.fill(kernel, 0);
		return this;
	}
	
	public ConvConf setKernelHeight (int kernelHeight) {
		this.kernel[0] = kernelHeight;
		return this;
	}
	
	public ConvConf setKernelWidth (int kernelWidth) {
		this.kernel[1] = kernelWidth;
		return this;
	}
	
	public ConvConf setKernel (int ndx, int val) {
		this.kernel[ndx] = val;
		return this;
	}
	
	public int getKernelSize () {
		return kernelSize;
	}
	
	public int getKernelHeight () {
		return kernel[0];
	}
	
	public int getKernelWidth () {
		return kernel[1];
	}
	
	public int getKernel (int ndx) {
		return kernel[ndx];
	}
	
	public ConvConf setStrideSize (int strideSize) {
		this.strideSize = strideSize;
		this.stride = new int [strideSize];
		Arrays.fill(stride, 0);
		return this;
	}
	
	public ConvConf setStrideHeight (int strideHeight) {
		this.stride[0] = strideHeight;
		return this;
	}
	
	public ConvConf setStrideWidth (int strideWidth) {
		this.stride[1] = strideWidth;
		return this;
	}
	
	public ConvConf setStride (int ndx, int val) {
		this.stride[ndx] = val;
		return this;
	}
	
	public int getStrideSize () {
		return strideSize;
	}
	
	public int getStrideHeight () {
		return stride[0];
	}
	
	public int getStrideWidth () {
		return stride[1];
	}
	
	public int getStride (int ndx) {
		return stride[ndx];
	}
	
	public ConvConf setNumberOfGroup (int groups) {
		this.groups = groups;
		return this;
	}
	
	public int numberOfGroups () {
		return groups;
	}
	
	public ConvConf setWeightsLearningRateMultiplier (float weightsLearningRateMultiplier) {
		this.weightsLearningRateMultiplier = weightsLearningRateMultiplier;
		return this;
	}
	
	public float getWeightsLearningRateMultiplier () {
		return weightsLearningRateMultiplier;
	}
	
	public ConvConf setBiasLearningRateMultiplier (float biasLearningRateMultiplier) {
		this.biasLearningRateMultiplier = biasLearningRateMultiplier;
		return this;
	}
	
	public float getBiasLearningRateMultiplier () {
		return biasLearningRateMultiplier;
	}
}
