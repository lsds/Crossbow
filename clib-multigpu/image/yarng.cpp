#include "yarng.h"

#include "../random/generator.hpp"

#include <mutex>

using namespace crossbow;

static CrossbowRandomGenerator *generator = NULL;
static int initialised = 0;

static std::mutex mtx;

void crossbowYarngInit (unsigned int seed) {
	if (initialised)
		return;
	generator = new CrossbowRandomGenerator (seed);
}

float crossbowYarngNext (float start, float end) {
	float value = 0;
    /* Lock */
	mtx.lock();
	generator->randomUniformFill (&value, 1, start, end);
    /* Unlock */
	mtx.unlock();
	return value;
}
