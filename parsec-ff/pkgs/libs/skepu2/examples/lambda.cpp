#include <iostream>
#include <skepu2.hpp>

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
	
	auto square = skepu2::Map<2>([](float a, float b) { return a * b; });
	square.setBackend(spec);
	
	auto dotprod = skepu2::MapReduce<2>(
		[](float a, float b) { return a * b; },
		[](float a, float b) { return a + b; }
	);
	
	
	auto sum =  skepu2::Reduce([](float a, float b) { return a + b; });
	
	auto sum2d =  skepu2::Reduce([](float a, float b) { return a + b; }, [](float a, float b) { return a + b; });
	
	auto cum_sum = skepu2::Scan([](float a, float b) { return a + b; });
	
	
	auto conv = skepu2::MapOverlap([](int o, size_t stride, const float *a)
	{
		const int s = (int)stride;
		float sum = 0;
		for (int i = -o; i <= o; i++)
			sum += a[i];
		return sum;
	});
	
	auto conv2d = skepu2::MapOverlap([](int ox, int oy, size_t stride, const float *m, const skepu2::Mat<float> filter)
	{
		float res = 0;
		for (int y = -oy; y <= oy; ++y)
			for (int x = -ox; x <= ox; ++x)
				res += m[y*(int)stride+x] * filter.data[(y+oy)*ox + (x+ox)];
		return res;
	});
	
	
	auto caller = skepu2::Call([](skepu2::Vec<float> r, float a, float b) { r[0] = a * b; });
	
	
	skepu2::Vector<float> v1(size, 3), v2(size, 7), r(size);
	skepu2::Matrix<float> m1(size, size, 3);
	std::cout << "v1: " << v1 << "\nv2: " << v2 << "\n";
	
	square(r, v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	square(r.begin(), r.end(), v1.begin(), v2.begin());
	std::cout << "Map: r = " << r << "\n";
	
	square(r.begin(), r.end(), v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	square(r, v1.begin(), v2);
	std::cout << "Map: r = " << r << "\n";
	
	
	std::cout << "MapReduce: dotprod(v1, v2) = " << dotprod(v1, v2) << "\n";
	
	std::cout << "Reduce 1D: sum(v1) = " << sum(v1) << "\n";
	
	std::cout << "Reduce 2D: sum2d(m1) = " << sum2d(m1) << "\n";
	
	cum_sum(r, v1);
	std::cout << "Scan: cum_sum(v1) = " << r << "\n";
	
	conv.setOverlap(2);
	conv.setEdgeMode(skepu2::Edge::Pad);
	conv.setPad(0);
	conv(r, v1);
	
	std::cout << "MapOverlap1D: conv(r, v1) = " << r << "\n";
	
	int o = 1;
	skepu2::Matrix<float> filter(2*o+1, 2*o+1, 1), m(size, size, 5), rm(size - 2*o, size - 2*o);
	
	conv2d.setOverlap(o);
	conv2d.setEdgeMode(skepu2::Edge::Pad);
	conv2d.setPad(0);
	conv2d(rm, m, filter);
	
	
	std::cout << "MapOverlap2D: conv2d(rm, m, filter) = " << rm << "\n";
	
	skepu2::Vector<float> ans(1);
	caller(ans, 4, 5);
	std::cout << "Call: caller(ans, 4, 5) = " << ans[0] << "\n";
	
	return 0;
}

