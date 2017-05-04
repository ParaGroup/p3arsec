#include <iostream>
#include <skepu2.hpp>


[[skepu::userconstant]]
constexpr size_t SIZE = 10;

template<typename T, typename V>
T arr(skepu2::Index1D row,  skepu2::Mat<T> m, const skepu2::Vec<T> v, skepu2::Vec<V> v2 [[skepu::out]])
{
	T res = 0;
	for (size_t i = 0; i < SIZE; ++i)
	{
		res = m.data[row.i * SIZE + i] * v.data[i];
	}
	v2.data[row.i * 3] = row.i;
	return res;
}


int main()
{
	auto mvprod = skepu2::Map<0>(arr<float, size_t>);
	
	skepu2::Vector<float> v0(SIZE);
	skepu2::Vector<size_t> v1(SIZE*3);
	skepu2::Matrix<float> m1(SIZE, SIZE);
	m1.randomize(3, 9);
	
	// Sets v0 = 1 2 3 4 5...
	for(int i = 0; i < SIZE; ++i)
		v0[i] = (float)(i+10);
	
//	std::cout<<"v0: " <<v0 <<"\n";
//	std::cout<<"m1: " <<m1 <<"\n";
	
	for (auto backend : skepu2::Backend::availableTypes())
	{
		std::cout << "---------[ " << backend << " ]---------\n";
		
		skepu2::Vector<float> r(SIZE);
		mvprod.setBackend(skepu2::BackendSpec{backend});
		
		auto s = mvprod(r, m1, v0, v1);
		std::cout << "CPU: " << s << "\n";
		std::cout << "r: " << r <<"\n";
		std::cout << "v1: " << v1 <<"\n";
		std::cout << "m1: " << m1 <<"\n";
	}
	
	return 0;
}
