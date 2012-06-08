/*
	(C) 2012 Jeff Chien
	
	Main file for the parallel diffuse-limited aggregation CUDA implementation.
 */

#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <ctime>

#include "common.h"
#include "cuda.h"
#include "file.h"
#include "gl.h"
#include "vec.cuh"

int main(int argc, char** argv)
{
	//*
	pdla::init_cuda();
	pdla::pdla_result_t p = pdla::run(16);
	std::cout << "Took " << p.elapsed << " seconds" << std::endl;
	if(!pdla::save_to_file(p, "result.csv"))
		std::cout << "Unable to save results";
	pdla::render(argc, argv, p);
	// */
	/*
	pdla::pdla_result_t p;
	pdla::load_from_file(p, "result.csv");
	// */
	return(0);
}