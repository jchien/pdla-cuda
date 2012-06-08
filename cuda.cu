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

CUDA_DEVICE_CALLABLE void pdla::vec::rand(curandState* rand, float minLen, float maxLen)
{
	float len = minLen + (maxLen - minLen) * curand_uniform(rand);
	float theta = TAU * curand_uniform(rand);
	x = len * cos(theta);
	y = len * sin(theta);
}

__global__ void kernel(context_t* ctx)
{
	int local_i		= threadIdx.x;
	int global_i	= blockIdx.x * blockDim.x + local_i;

	// Get which seed and particle we're comparing
#if PDLA_MODE
	int seed		= (int)((float)global_i * ctx->numSeeds / NUM_KERNELS);
	if(seed >= ctx->numSeeds)
		seed = ctx->numSeeds - 1;
	int part		= (int)(global_i - (float)seed * NUM_KERNELS / ctx->numSeeds);
	if(part >= NUM_KERNELS / ctx->numSeeds)
		return;
#else
	int seed		= global_i;
	if(seed >= ctx->numSeeds)
		return;
	int part		= 0;
#endif

	// PRNGs for the drifting particle
	curandState partRand;
	curand_init(part, ctx->numSeeds, 0, &partRand);

	// Setup particle data
	unsigned long long seedT = ctx->seedT[seed];
	pdla::vec seedPos(ctx->seedPos[seed]), oldPartPos, partPos, walk;
	oldPartPos.rand(&partRand, ctx->maxRadius + 4, ctx->maxRadius + 5);

#if KERNEL_DEBUG_1
	// Debug output
	ctx->debugArr[global_i].seed = seed;
	ctx->debugArr[global_i].part = part;
	ctx->debugArr[global_i].oldPart = oldPartPos;
#endif

	// Until someone collided with something
	for(unsigned long long time = 0; time < ctx->resTime; time++)
	{
		// 1) Drift particle
		walk.rand(&partRand, 0.4f, 0.6f);
		partPos = oldPartPos + walk;
		// Only check if the seed "existed then"
		if(time >= seedT)
		{
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
				// Rewind so that part is touching seed
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
		}
		// 4) Set up for next iteration (bounding box)
		oldPartPos = partPos;
		oldPartPos.boundingBox(ctx->maxRadius + 8);
	}
}

__global__ void addSeed(context_t* ctx)
{
	int minIndex = 0;
	for(int i = 1; i < ctx->resCount; i++)
		if(ctx->res[i].time < ctx->res[minIndex].time)
			minIndex = i;

	result_t res = ctx->res[minIndex];
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

	thrust::device_vector<context_t>	ctx(1);
	thrust::device_vector<debug_t>		debug(NUM_KERNELS);
	thrust::device_vector<result_t>		results(NUM_KERNELS);
	thrust::device_vector<vec>			seedPos(NUM_KERNELS + 1);
	thrust::device_vector<unsigned long long>
										seedT(NUM_KERNELS + 1);

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
	tmp.debugArr = debug.data().get();

	seedPos[0] = vec(0, 0);
	seedT[0] = 0;
	ctx[0] = tmp;

	clock_t t0 = clock();
	for(int i = 1; i < numSeeds; i++)
	{
		kernel<<<gridDim3, blockDim3>>>(ctx.data().get());
		
#if KERNEL_DEBUG_1
		{
			cudaThreadSynchronize();
			std::cout << "i: " << i << std::endl; 
			std::cout << "seeds: ";
			for(int j = 0; j < debug.size(); j++)
				std::cout << static_cast<debug_t>(debug[j]).seed << ", ";
			std::cout << std::endl;
			std::cout << "parts: ";
			for(int j = 0; j < debug.size(); j++)
				std::cout << static_cast<debug_t>(debug[j]).part << ", ";
			std::cout << std::endl;

			
			context_t c = ctx[0];
			if(c.resIndex < 0)
				std::cout << "resIndex: " << c.resIndex << std::endl;
			else
			{
				result_t res = static_cast<result_t>(results[c.resIndex]);
				std::cout << "result: " << res.pos << "(" << res.pos.len() << ") at time t = " << res.time << std::endl;
			}
			//*
			/*
			std::cout << "oldPart: ";
			for(int j = 0; j < debug.size(); j++)
				std::cout << static_cast<debug_t>(debug[j]).oldPart << ", ";
			std::cout << std::endl << std::endl;
			// */
		}
#endif

		addSeed<<<1, 1>>>(ctx.data().get());
		
#if KERNEL_DEBUG_2
		{
			cudaThreadSynchronize();
			context_t c = ctx[0];
			std::cout << "time: " << c.time << std::endl;
			std::cout << "maxR: " << c.maxRadius << std::endl;
			std::cout << "numSeeds: " << c.numSeeds << std::endl;
			/*
			std::cout << "seeds: ";
			for(int j = 0; j < c.numSeeds; j++)
				std::cout << static_cast<vec>(seedPos[j]) << " @ " << seedT[j] << ", ";
			std::cout << std::endl << std::endl;
			// */
		}
#endif
	}
	cudaThreadSynchronize();
#if KERNEL_DEBUG_3
	context_t c = ctx[0];
	std::cout << "seeds: ";
	for(int j = 0; j < c.numSeeds; j++)
		std::cout << static_cast<vec>(seedPos[j]) << " @ " << seedT[j] << ", ";
	std::cout << std::endl << std::endl;
#endif
	clock_t t1 = clock();

	pdla_result_t p = {std::vector<vec>(numSeeds), 
						std::vector<vec>(numSeeds),
						std::vector<unsigned long long>(numSeeds),
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