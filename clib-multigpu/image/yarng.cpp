#include "yarng.h"

#include "../random/generator.hpp"

using namespace crossbow;

static CrossbowRandomGenerator *generator = NULL;
static int initialised = 0;

void crossbowYarngInit (unsigned int seed) {
	if (initialised)
		return;
	generator = new CrossbowRandomGenerator (seed);
}

float crossbowYarngNext (float start, float end) {
	float value = 0;
	generator->randomUniformFill (&value, 1, start, end);
	return value;
}
