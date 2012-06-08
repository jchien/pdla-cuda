/*
	(C) 2012 Jeff Chien
	
	Save to and load from csv files
 */

#include "cuda.h"

namespace pdla
{
	int save_to_file(pdla_result_t p, const char* file);
	int load_from_file(pdla_result_t& p, const char* file);
}