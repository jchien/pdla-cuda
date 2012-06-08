/*
	(C) 2012 Jeff Chien
	
	Vector class.
 */

#include <iostream>
#include <cmath>

#include <curand_kernel.h>

#include "common.h"

#ifndef __PDLA_VEC__
#define __PDLA_VEC__
namespace pdla
{
	class vec
	{
	public:
		float x, y;

		CUDA_CALLABLE_MEMBER vec()
		{
			x = 0;
			y = 0;
		}
		CUDA_CALLABLE_MEMBER ~vec(){}
		CUDA_CALLABLE_MEMBER vec(float _x, float _y)
		{
			x = _x;
			y = _y;
		}
		CUDA_CALLABLE_MEMBER vec(vec const& other)
		{
			x = other.x;
			y = other.y;
		}
		CUDA_CALLABLE_MEMBER vec operator+(vec const& other) const
		{
			vec v(x + other.x, y + other.y);
			return(v);
		}
		CUDA_CALLABLE_MEMBER vec operator-(vec const& other) const
		{
			vec v(x - other.x, y - other.y);
			return(v);
		}
		CUDA_CALLABLE_MEMBER vec operator*(float a) const
		{
			vec v(x * a, y * a);
			return(v);
		}
		CUDA_CALLABLE_MEMBER vec& operator=(vec const& other)
		{
			x = other.x;
			y = other.y;
			return(*this);
		}
		CUDA_CALLABLE_MEMBER float dot(vec const& other) const
		{
			return(x * other.x + y * other.y);
		}
		CUDA_CALLABLE_MEMBER float len() const
		{
			return(sqrt(dot(*this)));
		}
		CUDA_CALLABLE_MEMBER void boundingBox(float max)
		{
			if(x > max)
				x -= 2 * max;
			else if(x < -max)
				x += 2 * max;
			if(y > max)
				y -= 2 * max;
			else if(y < -max)
				y += 2 * max;
		}
		CUDA_DEVICE_CALLABLE void rand(curandState* rand, float minLen, float maxLen);

		friend std::ostream& operator<< (std::ostream &out, vec &v);
	};
	std::ostream& operator<< (std::ostream &out, vec &v);
};
#endif