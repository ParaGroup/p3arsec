#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const std::string ScanKernel_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME_Scan(__global SKEPU_SCAN_TYPE* input, __global SKEPU_SCAN_TYPE* output, __global SKEPU_SCAN_TYPE* blockSums, size_t n, size_t numElements, __local SKEPU_SCAN_TYPE* sdata)
{
	const size_t threadIdx = get_local_id(0);
	const size_t blockDim = get_local_size(0);
	const size_t blockIdx = get_group_id(0);
	const size_t gridDim = get_num_groups(0);
	unsigned int pout = 0;
	unsigned int pin = 1;
	size_t mem = get_global_id(0);
	size_t gridSize = blockDim * gridDim;
	size_t numBlocks = numElements / blockDim + (numElements % blockDim == 0 ? 0:1);
	
	for (size_t blockNr = blockIdx; blockNr < numBlocks; blockNr += gridDim)
	{
		sdata[pout*n+threadIdx] = (mem < numElements) ? input[mem] : 0;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (size_t offset = 1; offset < n; offset *=2)
		{
			pout = 1-pout;
			pin = 1-pout;
			if (threadIdx >= offset)
				sdata[pout * n + threadIdx] = SKEPU_FUNCTION_NAME_SCAN(sdata[pin * n + threadIdx], sdata[pin * n + threadIdx - offset]);
			else
				sdata[pout * n + threadIdx] = sdata[pin*n+threadIdx];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
		if (threadIdx == blockDim - 1)
			blockSums[blockNr] = sdata[pout * n + blockDim - 1];
		
		if (mem < numElements)
			output[mem] = sdata[pout * n + threadIdx];
		mem += gridSize;
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
)~~~";

const std::string ScanUpdate_CL = R"~~~(
#define NULL 0
__kernel void SKEPU_KERNEL_NAME_ScanUpdate(__global SKEPU_SCAN_TYPE* data, __global SKEPU_SCAN_TYPE* sums, int isInclusive, SKEPU_SCAN_TYPE init, size_t n, __global SKEPU_SCAN_TYPE* ret, __local SKEPU_SCAN_TYPE* sdata)
{
	__local SKEPU_SCAN_TYPE offset;
	__local SKEPU_SCAN_TYPE inc_offset;
	const size_t threadIdx = get_local_id(0);
	const size_t blockDim = get_local_size(0);
	const size_t blockIdx = get_group_id(0);
	const size_t gridDim = get_num_groups(0);
	size_t gridSize = blockDim * gridDim;
	size_t mem = get_global_id(0);
	size_t numBlocks = n / blockDim + (n % blockDim == 0 ? 0:1);
	
	for (size_t blockNr = blockIdx; blockNr < numBlocks; blockNr += gridDim)
	{
		if (threadIdx == 0)
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
				if(blockNr > 0)
					offset = sums[blockNr-1];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (isInclusive == 1)
		{
			sdata[threadIdx] = (mem >= n)
				? 0 : (blockNr > 0)
					? SKEPU_FUNCTION_NAME_SCAN(offset, data[mem]) : data[mem];
			
			if(mem == n-1 && ret != NULL)
				*ret = sdata[threadIdx];
		}
		else
		{
			if(mem == n-1 && ret != NULL)
				*ret = SKEPU_FUNCTION_NAME_SCAN(inc_offset, data[mem]);
			
			sdata[threadIdx] = (threadIdx == 0)
				? offset : (mem-1 < n)
					? SKEPU_FUNCTION_NAME_SCAN(offset, data[mem-1]) : 0;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (mem < n)
			data[mem] = sdata[threadIdx];
		mem += gridSize;
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
)~~~";


const std::string ScanAdd_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME_ScanAdd(__global SKEPU_SCAN_TYPE* data, SKEPU_SCAN_TYPE sum, size_t n)
{
	size_t i = get_global_id(0);
	size_t gridSize = get_local_size(0) * get_num_groups(0);
	
	while (i < n)
	{
		data[i] = SKEPU_FUNCTION_NAME_SCAN(data[i], sum);
		i += gridSize;
	}
}
)~~~";


const std::string Constructor = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:
	
	enum
	{
		KERNEL_SCAN = 0,
		KERNEL_SCAN_UPDATE,
		KERNEL_SCAN_ADD,
		KERNEL_COUNT
	};
	
	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
	}
	
	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;
		
		std::string source = skepu2::backend::cl_helpers::replaceSizeT(R"###(SKEPU_OPENCL_KERNEL)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_scan = clCreateKernel(program, "SKEPU_KERNEL_NAME_Scan", &err);
			CL_CHECK_ERROR(err, "Error creating Scan kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			cl_kernel kernel_scan_update = clCreateKernel(program, "SKEPU_KERNEL_NAME_ScanUpdate", &err);
			CL_CHECK_ERROR(err, "Error creating Scan update kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			cl_kernel kernel_scan_add = clCreateKernel(program, "SKEPU_KERNEL_NAME_ScanAdd", &err);
			CL_CHECK_ERROR(err, "Error creating Scan add kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			kernels(counter, KERNEL_SCAN,        &kernel_scan);
			kernels(counter, KERNEL_SCAN_UPDATE, &kernel_scan_update);
			kernels(counter, KERNEL_SCAN_ADD,    &kernel_scan_add);
			counter++;
		}
		
		initialized = true;
	}
	
	static void scan
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *input, skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *output, skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *blockSums,
		size_t n, size_t numElements, size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums->getDeviceDataPointer(), n, numElements);
		clSetKernelArg(kernel, 5, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan kernel");
	}
	
	static void scanUpdate
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *data, skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *sums,
		int isInclusive, SKEPU_SCAN_TYPE init, size_t n,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *ret,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN_UPDATE);
		cl_mem retCL = (ret != nullptr) ? ret->getDeviceDataPointer() : NULL;
		skepu2::backend::cl_helpers::setKernelArgs(kernel, data->getDeviceDataPointer(), sums->getDeviceDataPointer(), isInclusive, init, n, retCL);
		clSetKernelArg(kernel, 6, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan update kernel");
	}
	
	static void scanAdd
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_SCAN_TYPE> *data,
		SKEPU_SCAN_TYPE sum, size_t n
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN_ADD);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, data->getDeviceDataPointer(), sum, n);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan add kernel");
	}
};
)~~~";


std::string createScanKernelProgram_CL(UserFunction &scanFunc, std::string dir)
{
	std::stringstream sourceStream;
	
	if (scanFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
	for (UserType *RefType : scanFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(scanFunc) << ScanKernel_CL << ScanUpdate_CL << ScanAdd_CL;
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_ScanKernel_" + scanFunc.uniqueName;
	const std::string className = "CLWrapperClass_" + kernelName;
	
	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_ScanType, scanFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_ScanFuncName, scanFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}
