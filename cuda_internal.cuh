/*
	(C) 2012 Jeff Chien
	
	CUDA internal macros and structures.
 */

#include "vec.cuh"

#ifndef	__PDLA_CUDA_INTERNAL__
#define	__PDLA_CUDA_INTERNAL__

#define	BLOCK_SIZE			1024
#define	GRID_SIZE			2
#define	NUM_KERNELS			(GRID_SIZE * BLOCK_SIZE)
typedef unsigned long long	pdla_time_t;

struct result_t
{
	unsigned long long time;
	pdla::vec pos;
};

struct context_t
{
	int randSeed;			// Random seed
	pdla_time_t time;		// Time step

	float maxRadius;		// Maximum seed radius of all of seedPos
	int numSeeds;			// Number of seeds
	pdla::vec* seedPos;		// Positions of seeds
	pdla_time_t* seedT;		// Time steps of seeds

	unsigned int resCount;	// Number of results
	pdla_time_t resTime;	// Time of earliest result
	result_t* res;			// Results
};

#endif