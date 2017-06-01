#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapReduceKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS SKEPU_REDUCE_RESULT_TYPE *output, size_t w, size_t n, size_t base)
{
	extern __shared__ alignas(SKEPU_REDUCE_RESULT_TYPE) char _sdata[];
	SKEPU_REDUCE_RESULT_TYPE *sdata = reinterpret_cast<SKEPU_REDUCE_RESULT_TYPE*>(_sdata);
	
	size_t blockSize = blockDim.x;
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x * blockSize + tid;
	size_t gridSize = blockSize * gridDim.x;
	SKEPU_REDUCE_RESULT_TYPE result = 0;
	
	if (i < n)
	{
		SKEPU_INDEX_INITIALIZER
		result = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		i += gridSize;
	}
	
	while (i < n)
	{
		SKEPU_INDEX_INITIALIZER
		SKEPU_MAP_RESULT_TYPE tempMap = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		result = SKEPU_FUNCTION_NAME_REDUCE(result, tempMap);
		i += gridSize;
	}
	
	sdata[tid] = result;
	__syncthreads();
	
	if (blockSize >= 1024) { if (tid < 512 && tid + 512 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >=  512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >=  256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >=  128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +  64]); } __syncthreads(); }
	if (blockSize >=   64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +  32]); } __syncthreads(); }
	if (blockSize >=   32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +  16]); } __syncthreads(); }
	if (blockSize >=   16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   8]); } __syncthreads(); }
	if (blockSize >=    8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   4]); } __syncthreads(); }
	if (blockSize >=    4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   2]); } __syncthreads(); }
	if (blockSize >=    2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   1]); } __syncthreads(); }
	
	if (tid == 0)
		output[blockIdx.x] = sdata[tid];
}
)~~~";

const char *ReduceKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_REDUCE_RESULT_TYPE *input, SKEPU_REDUCE_RESULT_TYPE *output, size_t n, size_t blockSize, bool nIsPow2)
{
	extern __shared__ alignas(SKEPU_REDUCE_RESULT_TYPE) char _sdata[];
	SKEPU_REDUCE_RESULT_TYPE* sdata = reinterpret_cast<SKEPU_REDUCE_RESULT_TYPE*>(_sdata);
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockSize*2 + threadIdx.x;
	size_t gridSize = blockSize*2*gridDim.x;
	SKEPU_REDUCE_RESULT_TYPE result = 0;
	
	if(i < n)
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
	// in a lParamer gridSize and therefore fewer elements per thread
	while(i < n)
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
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >=  512) { if (tid < 256) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >=  256) { if (tid < 128) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >=  128) { if (tid <  64) { sdata[tid] = result = SKEPU_FUNCTION_NAME_REDUCE(result, sdata[tid +  64]); } __syncthreads(); }
	
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


std::string createMapReduceKernelProgram_CU(UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapFuncParams;
	std::string indexInitializer;
	bool first = true;
	
	if (mapFunc.indexed1D)
	{
		SSMapFuncParams << "index";
		indexInitializer = "skepu2::Index1D index;\n\t\tindex.i = base + i;";
		first = false;
	}
	else if (mapFunc.indexed2D)
	{
		SSMapFuncParams << "index";
		indexInitializer = "skepu2::Index2D index;\n\t\tindex.row = (base + i) / w;\n\t\tindex.col = (base + i) % w;";
		first = false;
	}
	
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		SSMapFuncParams << param.name << "[i]";
		first = false;
	}
	
	for (UserFunction::RandomAccessParam& param : mapFunc.anyContainerParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.fullTypeName << " " << param.name << ", ";
		SSMapFuncParams << param.name;
		first = false;
	}
	
	for (UserFunction::Param& param : mapFunc.anyScalarParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSMapFuncParams << param.name;
		first = false;
	}
	
	const std::string kernelName = ResultName + "_MapReduceKernel_" + mapFunc.uniqueName + "_" + reduceFunc.uniqueName;
	const std::string reduceKernelName = kernelName + "_ReduceOnly";
	
	std::string kernelSource = MapReduceKernelTemplate_CU;
	replaceTextInString(kernelSource, PH_MapResultType, mapFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_ReduceResultType, reduceFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_MapFuncName, mapFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_ReduceFuncName, reduceFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(kernelSource, PH_MapParams, SSMapFuncParams.str());
	replaceTextInString(kernelSource, PH_IndexInitializer, indexInitializer);
	
	std::string reduceKernelSource = ReduceKernelTemplate_CU;
	replaceTextInString(reduceKernelSource, PH_ReduceResultType, reduceFunc.resolvedReturnTypeName);
	replaceTextInString(reduceKernelSource, PH_KernelName, reduceKernelName);
	replaceTextInString(reduceKernelSource, PH_ReduceFuncName, reduceFunc.funcNameCUDA());
	
	std::string totalSource = kernelSource + reduceKernelSource;
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << totalSource;
	return kernelName;
}