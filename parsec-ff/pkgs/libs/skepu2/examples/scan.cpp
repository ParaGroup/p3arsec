#include <iostream>
#include <skepu2.hpp>

float plus_f(float a, float b)
{
	return a + b;
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
	
	skepu2::Vector<float> v(size), r(size);
	v.randomize(0, 10);
	
	std::cout << "v: " << v << "\n";
	
	auto prefix_sum = skepu2::Scan(plus_f);
	prefix_sum.setBackend(spec);
	
	prefix_sum.setStartValue(10);
	prefix_sum.setScanMode(skepu2::ScanMode::Inclusive);
	
	// With containers
	prefix_sum(r, v);
	std::cout << "Scan inclusive: r = " << r << "\n";
	
	// With iterators
	prefix_sum(r.begin(), r.end(), v.begin());
	std::cout << "Scan inclusive: r = " << r << "\n";
	
	// With mixed iterators
	prefix_sum(r.begin(), r.end(), v);
	std::cout << "Scan inclusive: r = " << r << "\n";
	
	
	prefix_sum.setScanMode(skepu2::ScanMode::Exclusive);
	
	// With containers
	prefix_sum(r, v);
	std::cout << "Scan exclusive: r = " << r << "\n";
	
	// With iterators
	prefix_sum(r.begin(), r.end(), v.begin());
	std::cout << "Scan exclusive: r = " << r << "\n";
	
	// With mixed iterators
	prefix_sum(r, v.begin());
	std::cout << "Scan exclusive: r = " << r << "\n";
	
	return 0;
}

