#include <iostream>
#include <skepu2.hpp>


template<typename T>
T arr(skepu2::Index2D idx, const skepu2::Mat<T> lhs, const skepu2::Mat<T> rhs)
{
	T res = 0;
	for (size_t i = 0; i < lhs.cols; ++i)
		res += lhs.data[idx.row * lhs.cols + i] * rhs.data[i * rhs.cols + idx.col];
	return res;
}

// A helper function to calculate dense matrix-matrix product. Used to verify that the SkePU output is correct.
template<typename T>
void directMM(skepu2::Matrix<T> &lhs, skepu2::Matrix<T> &rhs, skepu2::Matrix<T> &res)
{
	int rows = lhs.total_rows();
	int cols = rhs.total_cols();
	
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < rows; ++j)
		{
			T sum{};
			for (int k = 0; k < cols; ++k)
				sum += lhs(i, k) * rhs(k, j);
			
			res(i, j) = sum;
		}
}

auto mmprod = skepu2::Map<0>(arr<float>);

void mmmult(skepu2::Matrix<float> &lhs, skepu2::Matrix<float> &rhs, skepu2::Matrix<float> &res, skepu2::BackendSpec *spec = nullptr)
{
	if (spec)
		mmprod.setBackend(*spec);
	
	mmprod(res, lhs, rhs);
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	size_t size = atoi(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
	
	skepu2::Matrix<float> lhs(size, size), rhs(size, size), res(size, size), res2(size, size);
	lhs.randomize(3, 9);
	rhs.randomize(0, 9);
	
	std::cout << "lhs: " << lhs << "\n";
	std::cout << "rhs: " << rhs << "\n";
	
	directMM(lhs, rhs, res2);
	mmmult(lhs, rhs, res, &spec);
	
	std::cout << "res: " << res << "\n";
	std::cout << "res2: " << res2 << "\n";
	
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			if (res(i, j) != res2(i, j))
				std::cout << "Output error at index (" << i << "," << j << "): " << res2[i] << " vs " << res[i] << "\n";
	
	return 0;
}
