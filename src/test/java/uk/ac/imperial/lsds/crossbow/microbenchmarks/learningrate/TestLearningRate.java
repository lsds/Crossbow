package uk.ac.imperial.lsds.crossbow.microbenchmarks.learningrate;

import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;

public class TestLearningRate {

	private static float getLearningRate(SolverConf conf, int id) {

		float rate;

		float base = conf.getBaseLearningRate();
		float gamma = conf.getGamma();
		float power = conf.getPower();
		float stepsize = conf.getStepSize();

		LearningRateDecayPolicy policy = conf.getLearningRateDecayPolicy();
		switch (policy) {

		case FIXED:
			rate = base;
			break;

		case INV:
			rate = base * (float) Math.pow(1F + gamma * ((float) (id + 1)), -power);
			break;

		case STEP:
			rate = base * (float) Math.pow(gamma, Math.floor((id + 1) / stepsize));
			break;

		case EXP:
			rate = base * (float) Math.pow(gamma, (id + 1));
			break;

		default:
			throw new IllegalArgumentException("error: invalid learning rate decay policy");
		}

		return rate;
	}

	public static void main(String[] args) {
		SolverConf conf = new SolverConf();
		conf.setLearningRateDecayPolicy(LearningRateDecayPolicy.INV).setBaseLearningRate(0.1F).setGamma(0.01F).setPower(0.2F);
		int tasksperepoch = 391;
		int epoch = 1;
		for (int id = 1; id <= 32000; ++id) {
			System.out.println(String.format("%06d\t%06d\t%7.5f", id, epoch, getLearningRate (conf, id)));
			if (id % tasksperepoch == 0)
				epoch++;
		}
	}
}
