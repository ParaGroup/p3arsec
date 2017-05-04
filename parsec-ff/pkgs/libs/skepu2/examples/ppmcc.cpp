/*!
 *  PPMCC stands for "Pearson product-moment correlation coefficient"
 *  In statistics, the Pearson coefficient of correlation is a measure by the
 *  linear dependence between two variables X and Y. The mathematical
 *  expression of the Pearson coefficient of correlation is as follows:
 *   r = ( (n*sum(X.Y)-sum(X)*sum(Y))/((n*sum(X^2)-(sum(X))^2)*(n*sum(Y^2)-(sum(Y))^2)) )
 */

#include <iostream>
#include <cmath>

#include <skepu2.hpp>

// Unary user-function used for mapping
template<typename T>
T square(T a)
{
	return a * a;
}

// Binary user-function used for mapping
template<typename T>
T mult(T a, T b)
{
	return a * b;
}

// User-function used for reduction
template<typename T>
T plus(T a, T b)
{
	return a + b;
}


using T = float;

// Skeleton definitions
auto sum = skepu2::Reduce(plus<T>);
auto dotProduct = skepu2::MapReduce<2>(mult<T>, plus<T>);
auto sumSquare = skepu2::MapReduce<1>(square<T>, plus<T>);

T ppmcc(skepu2::Vector<T> &x, skepu2::Vector<T> &y, skepu2::BackendSpec *spec = nullptr)
{
	if (spec)
	{
		sum.setBackend(*spec);
		dotProduct.setBackend(*spec);
		sumSquare.setBackend(*spec);
	}
	
	size_t N = x.size();
	T sumX = sum(x);
	T sumY = sum(y);
	
	return (N * dotProduct(x, y) - sumX * sumY)
		/ sqrt((N * sumSquare(x) - pow(sumX, 2)) * (N * sumSquare(y) - pow(sumY, 2)));
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " input_size backend\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
	
	// Vector operands
	skepu2::Vector<T> x(size), y(size);
	x.randomize(1, 3);
	y.randomize(2, 4);
	
	std::cout << "X: " << x << "\n";
	std::cout << "Y: " << y << "\n";
	
	T res = ppmcc(x, y, &spec);
	
	std::cout << "res: " << res << "\n";
	
	return 0;
}
