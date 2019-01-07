#ifndef __CROSSBOW_SOLVER_CONFIGURATION_H_
#define __CROSSBOW_SOLVER_CONFIGURATION_H_

#include "utils.h"

typedef struct crossbow_solver_configuration *crossbowSolverConfP;
typedef struct crossbow_solver_configuration {

	float alpha;
	int tau;

	crossbowLearningRateDecayPolicy_t learningRateDecayPolicy;
	float learningRate;

	double gamma;
	double power;

	int size;
	int numberofsteps;
	int *steps;
	int step;

	unsigned irregular;

	float momentum;
	crossbowMomentumMethod_t momentumMethod;
	
	float weightDecay;
	
	float baseModelMomentum;

	int superConvergence;

	float *circularLearningRate;
	float *circularMomentum;

    /* Not the most elegant solution,
     * but this flag signals the main 
     * thread (executioncontext.c) to 
     * copy the base model(s) to each
     * replica.
     *
     * It is triggered every time the
     * learning rate drops in a multi-
     * step configuration.
     */
	unsigned _copy;

} crossbow_solver_configuration_t;

crossbowSolverConfP crossbowSolverConfCreate ();

crossbowSolverConfP crossbowSolverConfReplicate (crossbowSolverConfP);

int crossbowSolverConfHasIrregularLearningRate (crossbowSolverConfP);

float crossbowSolverConfGetLearningRate (crossbowSolverConfP, int);

void crossbowSolverConfFree (crossbowSolverConfP);

#endif /* __CROSSBOW_SOLVER_CONFIGURATION_H_ */
