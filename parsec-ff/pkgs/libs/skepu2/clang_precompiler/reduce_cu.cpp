#include <algorithm>

#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

static const char *ReduceKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_REDUCE_RESULT_TYPE *input, SKEPU_REDUCE_RESULT_TYPE *output, size_t n, size_t blockSize, bool nIsPow2)
{
	extern __shared__ alignas(SKEPU_REDUCE_RESULT_TYPE) char _sdata[];
	SKEPU_REDUCE_RESULT_TYPE *sdata = reinterpret_cast<SKEPU_REDUCE_RESULT_TYPE*>(_sdata);
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
	size_t gridSize = blockSize * 2 * gridDim.x;
	
	SKEPU_REDUCE_RESULT_TYPE result = 0;
	
	if (i < n)
	{
		result = input[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		//This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (nIsPow2 || i + blockSize < n)
			result = SKEPU_FUNCTION_NAME_REDUCE(result, input[i+blockSize]);
		i += gridSize;
	}
	
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		result = SKEPU_FUNCTION_NAME_REDUCE(result, input[i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			result = SKEPU_FUNCTION_NAME_REDUCE(result, input[i+blockSize]);
		i += gridSize;
	}
	
	// each thread puts its local sum into shared memory
	sdata[tid] = result;
	__syncthreads();
	
	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid +  64]); } __syncthreads(); }
	
	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile SKEPU_REDUCE_RESULT_TYPE* smem = sdata;
		if (blockSize >=  64) { smem[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, smem[tid + 32]); }
		if (blockSize >=  32) { smem[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, smem[tid + 16]); }
		if (blockSize >=  16) { smem[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, smem[tid +  8]); }
		if (blockSize >=   8) { smem[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, smem[tid +  4]); }
		if (blockSize >=   4) { smem[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, smem[tid +  2]); }
		if (blockSize >=   2) { smem[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, smem[tid +  1]); }
	}
	
	// write result for this block to global mem
	if (tid == 0)
		output[blockIdx.x] = sdata[0];
}
)~~~";


std::string createReduce1DKernelProgram_CU(UserFunction &reduceFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_ReduceKernel_" + reduceFunc.uniqueName;
	
	std::string kernelSource = ReduceKernelTemplate_CU;
	replaceTextInString(kernelSource, PH_ReduceResultType, reduceFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_ReduceFuncName, reduceFunc.funcNameCUDA());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}


std::string createReduce2DKernelProgram_CU(UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_ReduceKernel_" + rowWiseFunc.uniqueName + "_" + colWiseFunc.uniqueName;
	
	std::string rowKernelSource = ReduceKernelTemplate_CU;
	replaceTextInString(rowKernelSource, PH_ReduceResultType, rowWiseFunc.resolvedReturnTypeName);
	replaceTextInString(rowKernelSource, PH_KernelName, kernelName + "_RowWise");
	replaceTextInString(rowKernelSource, PH_ReduceFuncName, rowWiseFunc.funcNameCUDA());
	
	std::string colKernelSource = ReduceKernelTemplate_CU;
	replaceTextInString(colKernelSource, PH_ReduceResultType, colWiseFunc.resolvedReturnTypeName);
	replaceTextInString(colKernelSource, PH_KernelName, kernelName + "_ColWise");
	replaceTextInString(colKernelSource, PH_ReduceFuncName, colWiseFunc.funcNameCUDA());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << rowKernelSource << colKernelSource;
	return kernelName;
}
