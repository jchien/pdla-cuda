/*
	(C) 2012 Jeff Chien
	
	CUDA functions.
 */

#include <vector>

#include "vec.cuh"

namespace pdla
{
#ifndef __PDLA_RESULT__
#define __PDLA_RESULT__
	struct pdla_result_t
	{
		std::vector<pdla::vec> pos;
		std::vector<unsigned long long> time;
		float elapsed;
	};
#endif

	void init_cuda();
	pdla_result_t run(int numSeeds);
};