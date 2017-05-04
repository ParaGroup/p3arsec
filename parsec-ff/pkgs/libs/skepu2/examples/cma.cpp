#include <iostream>
#include <skepu2.hpp>

template<typename T>
T sum(T a, T b)
{
	return a + b;
}

template<typename T, typename U>
U avg(skepu2::Index1D index, T sum)
{
	return (U)sum / (index.i + 1);
}


auto prefix_sum = skepu2::Scan(sum<int>);
auto average = skepu2::Map<1>(avg<int, float>);

void cma(skepu2::Vector<int> &in, skepu2::Vector<float> &out, skepu2::BackendSpec *spec = nullptr)
{
	if (spec)
	{
		prefix_sum.setBackend(*spec);
		average.setBackend(*spec);
	}
	
	prefix_sum(in, in);
	average(out, in);
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
	
	skepu2::Vector<int> in(size);
	skepu2::Vector<float> out(size);
	in.randomize(0, 10);
	
	if (size <= 50)
		std::cout << "Elements: " << in << "\n";
	
	cma(in, out, &spec);
	
	if (size <= 50)
		std::cout << "Cumulative moving average: " << out << "\n";
	else
		std::cout << "Average: " << out[size-1] << "\n";
	
	return 0;
}
