package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;

public class MatFactConf implements IConf {

    private int latents;

    private int rows;

    private int columns;

    private float lambda;

    private float learningRateEta0;

    private InitialiserConf modelVariableInitialiserConf;

    public MatFactConf () {

        modelVariableInitialiserConf = null;

        latents = 0;

        rows = 0;
        columns = 0;

        lambda = 0.1f;
        learningRateEta0 = 0.001f;
    }

    public InitialiserConf getModelVariableInitialiser () {
        return modelVariableInitialiserConf;
    }

    public MatFactConf setModelVariableInitialiser (InitialiserConf modelVariableInitialiserConf) {
        this.modelVariableInitialiserConf = modelVariableInitialiserConf;
        return this;
    }

    public int numberOfLatentVariables () {
        return latents;
    }

    public MatFactConf setNumberOfLatentVariables (int latents) {
        this.latents = latents;
        return this;
    }

    public int numberOfRows () {
        return rows;
    }

    public MatFactConf setNumberOfRows (int rows) {
        this.rows = rows;
        return this;
    }

    public int numberOfColumns () {
        return columns;
    }

    public MatFactConf setNumberOfColumns (int columns) {
        this.columns = columns;
        return this;
    }

    public float getLambda () { 
    	return lambda; 
    }

    public MatFactConf setLambda (float lambda) {
        this.lambda = lambda;
        return this;
    }

    public float getLearningRateEta0 () { return learningRateEta0; }

    public MatFactConf setLearningRateEta0 (float learningRateEta0) {
        this.learningRateEta0 = learningRateEta0;
        return this;
    }
}
