package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.BatchNormEstimatedMeanAndVarianceType;

public class BatchNormConf implements IConf {

	private int axis;

	private double epsilon;

	private double movingAverageFraction;
	
	private BatchNormEstimatedMeanAndVarianceType type;
	
	private boolean globalStatistics;
	
	private boolean bias;

	private InitialiserConf weightInitialiserConf, biasInitialiserConf;
	
	private float weightsLearningRateMultiplier, biasLearningRateMultiplier;

	public BatchNormConf () {

		axis = 1;
		
		epsilon = 0.00001D;
		movingAverageFraction = 0.9D;
		type = BatchNormEstimatedMeanAndVarianceType.FIXED;
		
		globalStatistics = true;
		bias = true;

		weightInitialiserConf = new InitialiserConf();
		biasInitialiserConf = new InitialiserConf();
		
		weightsLearningRateMultiplier = biasLearningRateMultiplier = 1;
	}
	
	public BatchNormConf setAxis (int axis) {
		this.axis = axis;
		return this;
	}

	public int getAxis () {
		return axis;
	}

	public BatchNormConf setEpsilon (double epsilon) {
		this.epsilon = epsilon;
		return this;
	}

	public double getEpsilon () {
		return epsilon;
	}
	
	public BatchNormConf setMovingAverageFraction (double movingAverageFraction) {
		this.movingAverageFraction = movingAverageFraction;
		return this;
	}
	
	public double getMovingAverageFraction () {
		return movingAverageFraction;
	}
	
	public BatchNormConf setEstimatedMeanAndVarianceType (BatchNormEstimatedMeanAndVarianceType type) {
		this.type = type;
		return this;
	}
	
	public BatchNormEstimatedMeanAndVarianceType getEstimatedMeanAndVarianceType () {
		return type;
	}
	
	public boolean hasBias () {
		return bias;
	}
	
	public BatchNormConf setBias (boolean bias) {
		this.bias = bias;
		return this;
	}
	
	public InitialiserConf getWeightInitialiser () {
		return weightInitialiserConf;
	}

	public BatchNormConf setWeightInitialiser (InitialiserConf weightInitialiserConf) {
		this.weightInitialiserConf = weightInitialiserConf;
		return this;
	}

	public InitialiserConf getBiasInitialiser () {
		return biasInitialiserConf;
	}

	public BatchNormConf setBiasInitialiser (InitialiserConf biasInitialiserConf) {
		this.biasInitialiserConf = biasInitialiserConf;
		return this;
	}

	public BatchNormConf setGlobalStatistics (boolean globalStatistics) {
		this.globalStatistics = globalStatistics;
		return this;
	}
	
	public boolean useGlobalStatistics () {
		return globalStatistics;
	}
	
	public BatchNormConf setWeightsLearningRateMultiplier (float weightsLearningRateMultiplier) {
		this.weightsLearningRateMultiplier = weightsLearningRateMultiplier;
		return this;
	}
	
	public float getWeightsLearningRateMultiplier () {
		return weightsLearningRateMultiplier;
	}
	
	public BatchNormConf setBiasLearningRateMultiplier (float biasLearningRateMultiplier) {
		this.biasLearningRateMultiplier = biasLearningRateMultiplier;
		return this;
	}
	
	public float getBiasLearningRateMultiplier () {
		return biasLearningRateMultiplier;
	}
}
