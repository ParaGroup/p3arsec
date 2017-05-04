#include <iostream>
#include <skepu2.hpp>

void swap_f(int *a, int *b)
{
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

void sort_f(skepu2::Vec<int> array, size_t nn)
{
#if SKEPU_USING_BACKEND_CL
	
	size_t idx = get_global_id(0);
	size_t l = nn / 2 + ((nn % 2 != 0) ? 1 : 0);
	
	for (size_t i = 0; i < l; ++i)
	{
		if (idx % 2 == 0 && idx < nn - 1 && array.data[idx] > array.data[idx + 1])
			swap_f(&array.data[idx], &array.data[idx + 1]);
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		if (idx % 2 == 1 && idx < nn - 1 && array.data[idx] > array.data[idx + 1])
			swap_f(&array.data[idx], &array.data[idx + 1]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
#else
	
	for (size_t c = 1; c <= nn - 1; c++)
		for (size_t d = c; d > 0 && array.data[d] < array.data[d-1]; --d)
			swap_f(&array.data[d], &array.data[d - 1]);
	
#endif
}


void sort(skepu2::Vector<int> &v, skepu2::BackendSpec spec)
{
	auto sort = skepu2::Call(sort_f);
	
	spec.setGPUBlocks(1);
	spec.setGPUThreads(v.size());
	sort.setBackend(spec);
	
	sort(v, v.size());
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	size_t size = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
	
	skepu2::Vector<int> v(size);
	v.randomize(0, 100);
	
	sort(v, spec);
	
	std::cout << v << "\n";
	
	return 0;
}

