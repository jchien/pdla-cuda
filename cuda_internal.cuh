/*
	(C) 2012 Jeff Chien
	
	CUDA internal macros and structures.
 */

#include "vec.cuh"

#define	BLOCK_SIZE		1024
#define	GRID_SIZE		2
#define	NUM_KERNELS		(GRID_SIZE * BLOCK_SIZE)

#define	KERNEL_DEBUG_1	0
#define	KERNEL_DEBUG_2	0
#define	KERNEL_DEBUG_3	0

// 0 = slow start (kernel = different seed, same particle)
// 1 = fast start (different seed, different particle)
#define	PDLA_MODE		0

#ifndef	__PDLA_CUDA_INTERNAL__
#define	__PDLA_CUDA_INTERNAL__

struct debug_t
{
};

struct result_t
{
	unsigned long long time;
	pdla::vec pos;
};

struct context_t
{
	int randSeed;					// Random seed
	unsigned long long time;		// Time step

	float maxRadius;				// Maximum seed radius of all of seedPos
	int numSeeds;					// Number of seeds
	pdla::vec* seedPos;				// Positions of seeds
	unsigned long long* seedT;		// Time steps of seeds

	unsigned int resCount;			// Number of results
	unsigned long long resTime;		// Time of earliest result
	result_t* res;					// Results

	debug_t* debugArr;				// Debug array
};

#endif