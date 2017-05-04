#include <iostream>
#include <utility>
#include <cfloat>

#include <milli.hpp>
#include <skepu2.hpp>

// User-function used for mapping
float mult_f_i(skepu2::Index1D index, float a, float b, skepu2::Vec<float> arr, float cons)
{
#if SKEPU_USING_BACKEND_CL
	// This code is not parsed by precompiler, but retained in output program for OpenCL backend. >>For debugging<<
//	printf("[OpenCL] index: %lu, a: %f, b: %f, arr[0]: %f, cons: %f\n", index.i, a, b, arr[0], cons);
#endif

#if SKEPU_USING_BACKEND_CUDA
	// This code is not parsed by precompiler, but retained in output program for CUDA backend. >>For debugging<<
//	printf("[CUDA] index: %lu, a: %f, b: %f, arr[0]: %f, cons: %f\n", index.i, a, b, arr[0], cons);
#endif

#if SKEPU_USING_BACKEND_CPU
	// This code is not parsed by precompiler, but retained in output program for CPU backend. >>For debugging<<
//	std::cout << "[CPU] index: " << index.i << " a: " << a << " b: " << b << " arr[0]: " << arr[0] << " cons: " << cons << "\n";
#endif

#if SKEPU_USING_BACKEND_OMP
	// This code is not parsed by precompiler, but retained in output program for OpenMP backend. >>For debugging<<
//	std::cout << "[OpenMP] index: " << index.i << " a: " << a << " b: " << b << " arr[0]: " << arr[0] << " cons: " << cons << "\n";
#endif

	return a*b + index.i + arr.data[0] * cons;
}

// User-function used for mapping
template<typename T>
T mult(T a, T b)
{
	return a*b;
}

// User-function used for reduction
template<typename T>
T plus(T a, T b)
{
	return a+b;
}

float testfn(float a, float b)
{
	return mult(a, b);
}


template<typename T>
T identity(T a)
{
	return a;
}

template<typename T>
T min(T a, T b)
{
	return a < b ? a : b;
}


auto modifiedDotProduct = skepu2::MapReduce<2>(mult_f_i, plus<float>);


int main(int argc, const char* argv[])
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size>\n";
		exit(1);
	}
	
	
	const size_t size = std::stoul(argv[1]);
	
	// Used for "elwise" container arguments
	skepu2::Vector<float> v0(size, 2.f);
	skepu2::Vector<float> v1(size, 5.f);
	
	// Used for "any" container argument (cf. SkePU 1 MapArray)
	skepu2::Vector<float> v2(1, 17.f);
	
	
	// Test MapReduce with two elwise operands, one array operand,
	// one constant scalar operand and with element indexing.
	
	for (auto backend : skepu2::Backend::availableTypes())
	{
		std::cout << "---------[ " << backend << " ]---------\n";
		
		for (int i = 0; i < 2; ++i)
		{
			modifiedDotProduct.setBackend(skepu2::BackendSpec{backend});
			milli::Reset();
			float r = modifiedDotProduct(v0, v1, v2, 5.f);
			double dur = milli::GetSeconds();
			
			std::cout << "    r: " << r / size << "\n";
			std::cout << "Time: " << dur << " seconds.\n\n";
		}
		
	}
	
	
	// Same computation, with iterator arguments instead.
	modifiedDotProduct.resetBackend();
	milli::Reset();
	float r = modifiedDotProduct(v0.begin(), v0.end(), v1.begin(), v2, 5.f);
	double dur = milli::GetSeconds();
	
	std::cout << "    r: " << r / size << "\n";
	std::cout << "Time: " << dur << " seconds.\n";
	
	
	// MapReduce with dense matrices
	auto sumOfProducts = skepu2::MapReduce<2>(mult<float>, plus<float>);
	
	auto sumOfProducts2 = skepu2::MapReduce<2>(mult<float>, testfn);
	
	skepu2::Matrix<float> m0(4, 4, 2.f);
	skepu2::Matrix<float> m1(4, 4, 5.f);
	
	milli::Reset();
	r = sumOfProducts(m0, m1);
	dur = milli::GetSeconds();
	
	std::cout << "    r: " << r << "\n";
	std::cout << "Time: " << dur << " seconds.\n";
	
	
	auto minCalc = skepu2::MapReduce<1>(identity<float>, min<float>);
	
	for (auto backend : skepu2::Backend::availableTypes())
	{
		std::cout << "---------[ " << backend << " ]---------\n";
		minCalc.setBackend(skepu2::BackendSpec{backend});
		minCalc.setStartValue(FLT_MAX);
		float r = minCalc(v0);
		
		std::cout << "min: " << r  << "\n";
	}
	
	return 0;
}

