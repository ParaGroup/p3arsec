/*
 * Taylor series calculation, natural log(1+x)  sum(1:N) (((-1)^(i+1))/i)*x^i
 */

#include <iostream>
#include <cmath>

#include <skepu2.hpp>


float nth_term(skepu2::Index1D index, float x)
{
	int k = index.i + 1;
	float temp_x = pow(x, k);
	int sign = (k % 2 == 0) ? -1 : 1;
	return sign * temp_x / k;
}

float plus(float a, float b)
{
	return a + b;
}

auto taylor = skepu2::MapReduce<0>(nth_term, plus);

float taylor_approx(float x, size_t N, skepu2::BackendSpec *spec = nullptr)
{
	
	taylor.setDefaultSize(N);
	if (spec) taylor.setBackend(*spec);
	
	return taylor(x);
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		std::cout << "Usage: " << argv[0] << " x-value number-of-terms backend\n";
		exit(1);
	}
	
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[3])};
	float x = atof(argv[1]);
	size_t N = std::stoul(argv[2]);
	
	std::cout << "Result: ln(" << x << ") = " << taylor_approx(x - 1, N, &spec) << "\n";
	
	return 0;
}