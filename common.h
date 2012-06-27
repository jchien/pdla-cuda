#define	PDLA_ASSERTS	1

#define	TAU		6.28318531f

#ifdef __CUDACC__
	#define CUDA_CALLABLE_MEMBER	__host__ __device__
	#define CUDA_HOST_CALLABLE		__host__
	#define CUDA_DEVICE_CALLABLE	__device__
#else
	#define CUDA_CALLABLE_MEMBER
	#define CUDA_HOST_CALLABLE
	#define CUDA_DEVICE_CALLABLE
#endif