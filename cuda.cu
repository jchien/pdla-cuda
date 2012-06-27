/*
	(C) 2012 Jeff Chien
	
	CUDA functions.
 */

#include <cmath>
#include <ctime>
#include <climits>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cutil_inline.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common.h"
#include "cuda.h"
#include "cuda_internal.cuh"
#include "vec.cuh"

static dim3 gridDim3(GRID_SIZE, 1, 1), blockDim3(BLOCK_SIZE, 1, 1);

// Initializes this vector to a random vector between minLen and maxLen (that has a random direction)
CUDA_DEVICE_CALLABLE void pdla::vec::rand(curandState* rand, float minLen, float maxLen)
{
	float len = minLen + (maxLen - minLen) * curand_uniform(rand);
	float theta = TAU * curand_uniform(rand);
	x = len * cos(theta);
	y = len * sin(theta);
}

__global__ void kernel(context_t* ctx)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= ctx->numSeeds)
		return;

	// PRNGs for the drifting particle
	curandState partRand;
	curand_init(ctx->randSeed, ctx->numSeeds, 0, &partRand);

	// Setup particle data
	pdla::vec seedPos(ctx->seedPos[index]), oldPartPos, partPos, walk;
	oldPartPos.rand(&partRand, ctx->maxRadius + 3, ctx->maxRadius + 4);
	
	// Until someone collided with something
	pdla_time_t time, resTime;
	time = ctx->time;
	// Cache resTime from the context so that we don't have to get it every time
	while(time < (resTime = ctx->resTime))
	{
		// Process in strides of 1024 to average out the cost of retrieving resTime
		// TODO: figure out correct stride length as function of seed count
		for(pdla_time_t maxTime = time + 1024; time < maxTime; time++)
		{
			// 1) Drift particle
			walk.rand(&partRand, 0.4f, 0.6f);
			partPos = oldPartPos + walk;
			// 2) Check collision
			pdla::vec seedVec = seedPos - oldPartPos;
			float t = seedVec.dot(walk) / walk.dot(walk);
			if(t < 0)
				t = 0;
			else if(t > 1)
				t = 1;
			// 3) If collision, set global vars (don't care about overwriting, but assign has to be atomic).
			float dist = (seedPos - (oldPartPos + walk * t)).len();
			if(dist < 2)
			{
				// Rewind so that particle is just touching seed
				float sintheta, costheta;
				__sincosf(acosf(seedVec.dot(walk) / (seedVec.len() * walk.len())),
							&sintheta, &costheta);
				float len = seedVec.len() * costheta - 
							sqrtf(4 - seedVec.dot(seedVec) * sintheta * sintheta);
				partPos = oldPartPos + walk * (len / walk.len());

				unsigned int index = atomicInc(&ctx->resCount, NUM_KERNELS);
				atomicCAS(&ctx->resTime, ULLONG_MAX, time);
				// Change the result
				ctx->res[index].time = time;
				ctx->res[index].pos = partPos;
			}
			// 4) Set up for next iteration (bounding box)
			oldPartPos = partPos;
			oldPartPos.boundingBox(ctx->maxRadius + 4);
		}
	}
}

__global__ void addSeed(context_t* ctx)
{
	// Get index of first result
	int minIndex = 0;
	for(int i = 1; i < ctx->resCount; i++)
		if(ctx->res[i].time < ctx->res[minIndex].time)
			minIndex = i;
	
	// Get the first result
	result_t res = ctx->res[minIndex];

	// Set context
	ctx->time = res.time + 1;
	if(res.pos.len() > ctx->maxRadius)
		ctx->maxRadius = res.pos.len();
	ctx->resCount = 0;
	ctx->resTime = ULLONG_MAX;
	ctx->seedPos[ctx->numSeeds] = res.pos;
	ctx->seedT[ctx->numSeeds] = res.time;
	ctx->numSeeds++;
}

void pdla::init_cuda()
{
	cutilSafeCall(cudaSetDevice(0));
}

pdla::pdla_result_t pdla::run(int numSeeds)
{
	if(numSeeds > NUM_KERNELS)
	{
		std::cerr << "Cannot have more seeds (" << numSeeds << ") than kernels (" << NUM_KERNELS << ")!" << std::endl;
		abort();
	}

	// Initialize device memory
	thrust::device_vector<context_t>	ctx(1);
	thrust::device_vector<result_t>		results(NUM_KERNELS);
	thrust::device_vector<vec>			seedPos(NUM_KERNELS + 1);
	thrust::device_vector<pdla_time_t>	seedT(NUM_KERNELS + 1);

	// Initialize context
	// Note: (thrust device vector).data().get() gets the pointer to device memory
	context_t tmp;
	tmp.randSeed = time(0);
	tmp.time = 0;
	tmp.maxRadius = 0;
	tmp.numSeeds = 1;
	tmp.resCount = 0;
	tmp.resTime = ULLONG_MAX; 
	tmp.seedPos = seedPos.data().get();
	tmp.seedT = seedT.data().get();
	tmp.res = results.data().get();

	seedPos[0] = vec(0, 0);
	seedT[0] = 0;
	ctx[0] = tmp;
	
	// Take time
	clock_t t0 = clock();
	
	// Send instructions to GPU
	for(int i = 1; i < numSeeds; i++)
	{
		kernel<<<gridDim3, blockDim3>>>(ctx.data().get());
		addSeed<<<1, 1>>>(ctx.data().get());
	}
	// Wait for execution to finish
	cudaThreadSynchronize();
	
	// Take time again
	clock_t t1 = clock();

	// Copy results from Thrust containers to STL containers
	pdla_result_t p = {std::vector<vec>(numSeeds), 
						std::vector<pdla_time_t>(numSeeds),
						1.0f * (t1 - t0) / CLOCKS_PER_SEC};
	for(int i = 0; i < numSeeds; i++)
	{
		p.pos[i] = seedPos[i];
		p.time[i] = seedT[i];
	}
	return(p);
}

std::ostream& pdla::operator<< (std::ostream &out, vec &v)
{
	out << "<" << v.x << ", " << v.y << ">";
	return(out);
}