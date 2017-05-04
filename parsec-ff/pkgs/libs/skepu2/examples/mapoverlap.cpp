#include <iostream>

#include <skepu2.hpp>

float over_1d(int, size_t stride, const float *a, int scale)
{
	const int s = (int)stride;
	return (a[-2*s]*4 + a[-1*s]*2 + a[0]*1 + a[1*s]*2 + a[2*s]*4) / scale;
}

float over_2d(int ox, int oy, size_t stride, const float *m, const skepu2::Mat<float> filter)
{
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; ++x)
			res += m[y*(int)stride+x] * filter.data[(y+oy)*ox + (x+ox)];
	return res;
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
	
	auto conv = skepu2::MapOverlap(over_1d);
	conv.setBackend(spec);
	conv.setOverlap(2);
	
	skepu2::Vector<float> v(size, 10), rv(size);
	
	std::cout << "v: " << v <<"\n";
	
	conv.setEdgeMode(skepu2::Edge::Cyclic);
	conv(rv, v, 13);
	std::cout << "Vector Cyclic:    rv = " << rv << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Duplicate);
	conv(rv, v, 13);
	std::cout << "Vector Duplicate  rv = " << rv << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Pad);
	conv.setPad(0);
	conv(rv, v, 13);
	std::cout << "Vector Pad 0:     rv = " << rv << "\n";
	
	
	
	skepu2::Matrix<float> m(size, size, 10), rm(size, size);
	
	
	int i = 0;
	for (int y = 0; y < size; ++y)
		for (int x = 0; x < size; ++x)
			m(y, x) = i++;
	
	std::cout << "m: " << m <<"\n";
	conv.setOverlap(2);
	conv.setOverlapMode(skepu2::Overlap::RowWise);
	
	conv.setEdgeMode(skepu2::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Row-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Row-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Row-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv.setOverlap(2);
	conv.setOverlapMode(skepu2::Overlap::ColWise);
	
	conv.setEdgeMode(skepu2::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Col-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Col-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Col-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv.setOverlap(2);
	conv.setOverlapMode(skepu2::Overlap::RowColWise);
	
	conv.setEdgeMode(skepu2::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Row-col-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Row-col-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Row-col-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv.setOverlap(2);
	conv.setOverlapMode(skepu2::Overlap::ColRowWise);
	
	conv.setEdgeMode(skepu2::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Col-row-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Col-row-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu2::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Col-row-wise Pad 0:     rm = " << rm << "\n";
	
	
	auto conv2 = skepu2::MapOverlap(over_2d);
	conv2.setBackend(spec);
	conv2.setOverlap(1, 1);
	conv2.setEdgeMode(skepu2::Edge::Pad);
	conv2.setPad(0);
	
	skepu2::Matrix<float> filter(2*1+1, 2*1+1, 1), rm2(size - 2*1, size - 2*1);
	
	conv2(rm2, m, filter);
	std::cout << "Matrix 2D Pad 0:    rm = " << rm2 << "\n";
	
	return 0;
}

