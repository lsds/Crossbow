#include "generator.hpp"

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/normal.hpp>

#include <boost/random/variate_generator.hpp>

namespace crossbow {

CrossbowRandomGenerator::CrossbowRandomGenerator (unsigned int seed) {

	this->seed = seed;
	this->rng  = new rng_t(seed);
}

float CrossbowRandomGenerator::nextafter (const float value) {

	return boost::math::nextafter<float>(value, std::numeric_limits<float>::max());
}

void CrossbowRandomGenerator::randomUniformFill (float *buffer, const int count, const float start, const float end) {

	if (buffer == NULL) {
		fprintf(stderr, "error: buffer to fill must not be null\n");
		exit (1);
	}

	if (count < 0) {
		fprintf(stderr, "error: number of buffer elements to fill must be greater or equal to 0\n");
		exit (1);
	}

	if (start > end) {
		fprintf(stderr, "error: invalid uniform random distribution specification\n");
		exit (1);
	}

	boost::uniform_real<float> dist (start, nextafter (end));
	boost::variate_generator<crossbow::rng_t *, boost::uniform_real<float> > variate_generator (this->rng, dist);

	for (int i = 0; i < count; ++i)
		buffer [i] = variate_generator ();

	return;
}

void CrossbowRandomGenerator::randomGaussianFill (float *buffer, const int count, const float mean, const float std, const int truncate) {

	if (buffer == NULL) {
		fprintf(stderr, "error: buffer to fill must not be null\n");
		exit (1);
	}

	if (count < 0) {
		fprintf(stderr, "error: number of buffer elements to fill must be greater or equal to 0\n");
		exit (1);
	}

	if (std <= 0) {
		fprintf(stderr, "error: invalid normal distribution specification\n");
		exit (1);
	}

	boost::normal_distribution<float> dist (mean, std);
	boost::variate_generator<crossbow::rng_t *, boost::normal_distribution<float> > variate_generator (this->rng, dist);
	
	/* float checksum = 0; */

	if (truncate) {

		/*
		 * Added on 14 Apr 2018: Hard-coded truncated version:
		 *
		 * Values whose magnitude is more than 2 standard deviations
		 * from the mean are dropped and re-picked.
		 */
		float min = mean - (2 * std);
		float max = mean + (2 * std);
		int maxiterations = 100;
	
		float sample;
		int correct = 0;
		for (int i = 0; i < count; ++i) {
			for (int j = 0; j < maxiterations; ++j) {
				sample = variate_generator ();
				if ((sample > min) && (sample < max)) {
					correct ++;
					break;
				}
			}
			buffer [i] = sample;
			/* checksum += buffer [i]; */
		}
		if (correct < count)
			fprintf(stderr, "warning: only %d out of %d values truncated\n", correct, count);
	}
	else {
		for (int i = 0; i < count; ++i) {
			buffer [i] = variate_generator ();
			/* checksum += buffer [i]; */
		}
	}
	/*
	 * fprintf(stdout, "[DBG] checksum is %.5f\n", checksum);
	 * fflush (stdout);
	 */

	return;
}

void CrossbowRandomGenerator::dump () {

	fprintf(stdout, "CrossbowRandom (%du)\n", seed);
	fflush (stdout);
}

}  /* namespace crossbow */
