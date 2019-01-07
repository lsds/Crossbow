package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;

public class InnerProductConf implements IConf {
	
	private int outputs;
	
	private boolean bias;
	
	private int axis;
	
	private InitialiserConf weightInitialiserConf, biasInitialiserConf;
	
	private float weightsLearningRateMultiplier, biasLearningRateMultiplier;
	
	public InnerProductConf() {
		outputs = 1;
		bias = true;
		axis = 1;
		weightInitialiserConf = new InitialiserConf();
		biasInitialiserConf = new InitialiserConf();
		weightsLearningRateMultiplier = biasLearningRateMultiplier = 1;
	}
	
	public int numberOfOutputs () {
		return outputs;
	}

	public InnerProductConf setNumberOfOutputs (int outputs) {
		this.outputs = outputs;
		return this;
	}
	
	public boolean hasBias () {
		return bias;
	}

	public InnerProductConf setBias (boolean bias) {
		this.bias = bias;
		return this;
	}
	
	public InitialiserConf getWeightInitialiser () {
		return weightInitialiserConf;
	}

	public InnerProductConf setWeightInitialiser (InitialiserConf weightInitialiserConf) {
		this.weightInitialiserConf = weightInitialiserConf;
		return this;
	}
	
	public InitialiserConf getBiasInitialiser () {
		return biasInitialiserConf;
	}

	public InnerProductConf setBiasInitialiser (InitialiserConf biasInitialiserConf) {
		this.biasInitialiserConf = biasInitialiserConf;
		return this;
	}

	public int getAxis () {
		return axis;
	}

	public InnerProductConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}
	
	public InnerProductConf setWeightsLearningRateMultiplier (float weightsLearningRateMultiplier) {
		this.weightsLearningRateMultiplier = weightsLearningRateMultiplier;
		return this;
	}
	
	public float getWeightsLearningRateMultiplier () {
		return weightsLearningRateMultiplier;
	}
	
	public InnerProductConf setBiasLearningRateMultiplier (float biasLearningRateMultiplier) {
		this.biasLearningRateMultiplier = biasLearningRateMultiplier;
		return this;
	}
	
	public float getBiasLearningRateMultiplier () {
		return biasLearningRateMultiplier;
	}
}
