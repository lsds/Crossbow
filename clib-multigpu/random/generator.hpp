#ifndef __CROSSBOW_RANDOM_GENERATORH_
#define __CROSSBOW_RANDOM_GENERATORH_

#include <boost/random.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <boost/math/special_functions/next.hpp>

namespace crossbow {

	typedef boost::mt19937 rng_t;

	class CrossbowRandomGenerator {

	public:
		CrossbowRandomGenerator (unsigned int);

		void randomUniformFill  (float *, const int, const float, const float);
		void randomGaussianFill (float *, const int, const float, const float, const int);

		void dump ();

	private:

	protected:
		float nextafter (const float);

		unsigned int seed;
		rng_t *rng;
	};

} /* namespace crossbow */

#endif /* __CROSSBOW_RANDOM_GENERATORH_ */
