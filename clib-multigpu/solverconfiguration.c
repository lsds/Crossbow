#include "solverconfiguration.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

#include <math.h>

crossbowSolverConfP crossbowSolverConfCreate () {
	crossbowSolverConfP p;

	p = (crossbowSolverConfP) crossbowMalloc (sizeof(crossbow_solver_configuration_t));
	/* Initialise values */
	p->alpha = 0.5;
	p->tau = 1;

	p->learningRateDecayPolicy = FIXED;
	p->learningRate = 0;

	p->gamma = 0;
	p->power = 0;

	p->size = 0;

	p->numberofsteps = 0;
	p->steps = NULL;
	p->step = 0;

	p->irregular = 0;

	p->momentum = 0;
	p->momentumMethod = POLYAK;
	p->weightDecay = 0;
	
	p->baseModelMomentum = 0;

	p->superConvergence = 0;

	p->circularLearningRate = NULL;
	p->circularMomentum = NULL;

	p->_copy = 0;

	return p;
}

crossbowSolverConfP crossbowSolverConfReplicate (crossbowSolverConfP conf) {
	int i;
	crossbowSolverConfP p;

	p = (crossbowSolverConfP) crossbowMalloc (sizeof(crossbow_solver_configuration_t));

	/* Initialise values */
	p->alpha = conf->alpha;
	p->tau = conf->tau;

	p->learningRateDecayPolicy = conf->learningRateDecayPolicy;
	p->learningRate = conf->learningRate;

	p->gamma = conf->gamma;
	p->power = conf->power;

	p->size = conf->size;

	p->numberofsteps = conf->numberofsteps;
	if (conf->steps) {
		p->steps = crossbowMalloc (p->numberofsteps * sizeof(int));
		for (i = 0; i < p->numberofsteps; ++i)
			p->steps[i] = conf->steps[i];
	} else {
		p->steps = NULL;
	}
	p->step = conf->step;

	p->irregular = conf->irregular;

	p->momentum = conf->momentum;
	p->momentumMethod = conf->momentumMethod;
	
	p->weightDecay = conf->weightDecay;
	
	p->baseModelMomentum = conf->baseModelMomentum;

	p->superConvergence = conf->superConvergence;
	
	if (conf->circularLearningRate) {
		p->circularLearningRate = (float *) crossbowMalloc (3 * sizeof(float));
		for (i = 0; i < 3; ++i)
			p->circularLearningRate [i] = conf->circularLearningRate [i];
	} else {
		p->circularLearningRate = NULL;
	}
	
	if (conf->circularMomentum) {
		p->circularMomentum =     (float *) crossbowMalloc (3 * sizeof(float));
		for (i = 0; i < 3; ++i)
			p->circularMomentum [i] = conf->circularMomentum [i];
	} else {
		p->circularMomentum = NULL;
	}

	p->_copy = conf->_copy;

	return p;
}

int crossbowSolverConfHasIrregularLearningRate (crossbowSolverConfP p) {
	return (p->irregular > 0);
}

float crossbowSolverConfGetLearningRate (crossbowSolverConfP p, int task) {
	float rate;
	switch (p->learningRateDecayPolicy) {
	case FIXED:
		rate = p->learningRate;
		break;
	case INV:
		rate = p->learningRate * (float) (pow (1.0 + p->gamma * ((double) (task + 1)), -(p->power)));
		break;
	case STEP:
		rate = p->learningRate * (float) (pow (p->gamma, floor ((task + 1) / p->size)));
		break;
	case MULTISTEP:
		if ((p->step < p->numberofsteps) && ((task + 1) >= p->steps[p->step])) {
			p->step ++;
			//if (task >= 62560) {
			//	p->gamma = 0.1;
			info("Changing learning rate to %.10f\n", p->learningRate * (float) (pow (p->gamma, p->step)));
			//}
            /* Signal copy of base model(s) to replicas */
			p->_copy = 1;
		}
		rate = p->learningRate * (float) (pow (p->gamma, p->step));
		break;
	case EXP:
		rate = p->learningRate * (float) (pow (p->gamma, (task + 1)));
		break;
	case CLR:
		unsupportedOperationException();
		break;
	default:
		err("invalid learning rate policy type\n");
	}
	return rate;
}

void crossbowSolverConfFree (crossbowSolverConfP p) {
	if (! p)
		return;
	if (p->steps)
		crossbowFree (p->steps, (p->numberofsteps * sizeof (int)));

	if (p->circularLearningRate)
		crossbowFree (p->circularLearningRate, 3 * sizeof(float));

	if (p->circularMomentum)
		crossbowFree (p->circularMomentum, 3 * sizeof(float));

	crossbowFree(p, sizeof(crossbow_solver_configuration_t));
}
