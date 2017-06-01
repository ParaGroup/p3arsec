#include <iostream>
#include <skepu2.hpp>

int mult_f(int a, int b)
{
	return a * b;
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
	
	auto square = skepu2::Map<2>(mult_f);
	square.setBackend(spec);
	
	skepu2::Vector<int> v1(size, 3), v2(size, 7), r(size);
	std::cout << "v1: " << v1 << "\nv2: " << v2 << "\n";
	
	square(r, v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	square(r.begin(), r.end(), v1.begin(), v2.begin());
	std::cout << "Map: r = " << r << "\n";
	
	square(r.begin(), r.end(), v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	square(r, v1.begin(), v2);
	std::cout << "Map: r = " << r << "\n";
	
	return 0;
}

