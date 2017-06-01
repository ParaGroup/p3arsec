#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const std::string ScanKernel_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME_ScanKernel(SKEPU_SCAN_TYPE* input, SKEPU_SCAN_TYPE* output, SKEPU_SCAN_TYPE* blockSums, size_t n, size_t numElements)
{
	extern __shared__ char _sdata[];
	SKEPU_SCAN_TYPE* sdata = reinterpret_cast<SKEPU_SCAN_TYPE*>(_sdata);
	
	size_t thid = threadIdx.x;
	unsigned int pout = 0;
	unsigned int pin = 1;
	size_t mem = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;
	size_t numBlocks = numElements / (blockDim.x) + (numElements % (blockDim.x) == 0 ? 0:1);
	size_t offset;
	
	for (size_t blockNr = blockIdx.x; blockNr < numBlocks; blockNr += gridDim.x)
	{
		sdata[pout*n+thid] = (mem < numElements) ? input[mem] : 0;
		
		__syncthreads();
		
		for (offset = 1; offset < n; offset *=2)
		{
			pout = 1-pout;
			pin = 1-pout;
			if (thid >= offset)
				sdata[pout*n+thid] = SKEPU_FUNCTION_NAME_SCAN(sdata[pin*n+thid], sdata[pin*n+thid-offset]);
			else
				sdata[pout*n+thid] = sdata[pin*n+thid];
			
			__syncthreads();
		}
		
		if (thid == blockDim.x - 1)
			blockSums[blockNr] = sdata[pout*n+blockDim.x-1];
		
		if (mem < numElements)
			output[mem] = sdata[pout*n+thid];
		
		mem += gridSize;
		
		__syncthreads();
	}
}
)~~~";

const std::string ScanUpdate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME_ScanUpdate(SKEPU_SCAN_TYPE *data, SKEPU_SCAN_TYPE *sums, int isInclusive, SKEPU_SCAN_TYPE init, size_t n, SKEPU_SCAN_TYPE *ret)
{
	extern __shared__ char _sdata[];
	SKEPU_SCAN_TYPE* sdata = reinterpret_cast<SKEPU_SCAN_TYPE*>(_sdata);
	
	__shared__ SKEPU_SCAN_TYPE offset;
	__shared__ SKEPU_SCAN_TYPE inc_offset;
	
	size_t thid = threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;
	size_t mem = blockIdx.x * blockDim.x + threadIdx.x;
	size_t numBlocks = n / (blockDim.x) + (n % (blockDim.x) == 0 ? 0:1);
	
	for (size_t blockNr = blockIdx.x; blockNr < numBlocks; blockNr += gridDim.x)
	{
		if (thid == 0)
		{
			if (isInclusive == 0)
			{
				offset = init;
				if (blockNr > 0)
				{
					offset = SKEPU_FUNCTION_NAME_SCAN(offset, sums[blockNr-1]);
					inc_offset = sums[blockNr-1];
				}
			}
			else
			{
				if (blockNr > 0)
					offset = sums[blockNr-1];
			}
		}
		
		__syncthreads();
		
		if (isInclusive == 1)
		{
			if (blockNr > 0)
				sdata[thid] = (mem < n) ? SKEPU_FUNCTION_NAME_SCAN(offset, data[mem]) : 0;
			else
				sdata[thid] = (mem < n) ? data[mem] : 0;
			
			if (mem == n-1)
				*ret = sdata[thid];
		}
		else
		{
			if (mem == n-1)
				*ret = SKEPU_FUNCTION_NAME_SCAN(inc_offset, data[mem]);
			
			if (thid == 0)
				sdata[thid] = offset;
			else
				sdata[thid] = (mem-1 < n) ? SKEPU_FUNCTION_NAME_SCAN(offset, data[mem-1]) : 0;
		}
		
		__syncthreads();
		
		if (mem < n)
			data[mem] = sdata[thid];
		
		mem += gridSize;
		
		__syncthreads();
	}
}
)~~~";


const std::string ScanAdd_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME_ScanAdd(SKEPU_SCAN_TYPE *data, SKEPU_SCAN_TYPE sum, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;
	
	while (i < n)
	{
		data[i] = SKEPU_FUNCTION_NAME_SCAN(data[i], sum);
		i += gridSize;
	}
}
)~~~";


std::string createScanKernelProgram_CU(UserFunction &scanFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_Scan_" + scanFunc.uniqueName;
	
	std::string kernelSource = ScanKernel_CU + ScanUpdate_CU + ScanAdd_CU;
	replaceTextInString(kernelSource, PH_ScanType, scanFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_ScanFuncName, scanFunc.funcNameCUDA());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}
