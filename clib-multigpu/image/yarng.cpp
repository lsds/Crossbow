#include "yarng.h"

#include "../random/generator.hpp"

using namespace crossbow;

static CrossbowRandomGenerator *generator = NULL;

void crossbowYarngInit (unsigned int seed) {
	generator = new CrossbowRandomGenerator (seed);
}

float crossbowYarngNext (float start, float end) {
	float value = 0;
	generator->randomUniformFill (&value, 1, start, end);
	return value;
}
