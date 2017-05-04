#include <algorithm>

#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

static const char *ReduceKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(__global SKEPU_REDUCE_RESULT_TYPE* input, __global SKEPU_REDUCE_RESULT_TYPE* output, size_t n, __local SKEPU_REDUCE_RESULT_TYPE* sdata)
{
	size_t blockSize = get_local_size(0);
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * blockSize + get_local_id(0);
	size_t gridSize = blockSize*get_num_groups(0);
	SKEPU_REDUCE_RESULT_TYPE result;
	
	if (i < n)
	{
		result = input[i];
		i += gridSize;
	}
	
	while (i < n)
	{
		result = SKEPU_FUNCTION_NAME_REDUCE(result, input[i]);
		i += gridSize;
	}
	
	sdata[tid] = result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (blockSize >= 1024) { if (tid < 512 && tid + 512 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = SKEPU_FUNCTION_NAME_REDUCE(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }
	
	if (tid == 0)
		output[get_group_id(0)] = sdata[tid];
}
)~~~";


const std::string Constructor1D = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:
	
	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
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
			cl_kernel kernel = clCreateKernel(program, "SKEPU_KERNEL_NAME", &err);
			CL_CHECK_ERROR(err, "Error creating map kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			kernels(counter++, &kernel);
		}
		
		initialized = true;
	}
	
	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		skepu2::backend::cl_helpers::setKernelArgs(kernels(deviceID), input, output, n);
		clSetKernelArg(kernels(deviceID), 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
};
)~~~";


std::string createReduce1DKernelProgram_CL(UserFunction &reduceFunc, std::string dir)
{
	std::stringstream sourceStream;
	
	if (reduceFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	for (UserType *RefType : reduceFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + reduceFunc.uniqueName;
	const std::string className = "CLWrapperClass_" + kernelName;
	
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(reduceFunc) << ReduceKernelTemplate_CL;
	
	std::string finalSource = Constructor1D;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_ReduceResultType, reduceFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_ReduceFuncName, reduceFunc.uniqueName);
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}



const std::string Constructor2D = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:
	
	enum
	{
		KERNEL_ROWWISE = 0,
		KERNEL_COLWISE,
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

			cl_kernel rowwisekernel = clCreateKernel(program, "SKEPU_KERNEL_NAME_RowWise", &err);
			CL_CHECK_ERROR(err, "Error creating row-wise Reduce kernel 'SKEPU_KERNEL_NAME'");
			
			cl_kernel colwisekernel = clCreateKernel(program, "SKEPU_KERNEL_NAME_ColWise", &err);
			CL_CHECK_ERROR(err, "Error creating col-wise Reduce kernel 'SKEPU_KERNEL_NAME'");
			
			kernels(counter, KERNEL_ROWWISE, &rowwisekernel);
			kernels(counter, KERNEL_COLWISE, &colwisekernel);
			counter++;
		}
		
		initialized = true;
	}
	
	static void reduceRowWise(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_ROWWISE);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input, output, n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
	
	static void reduceColWise(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_COLWISE);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input, output, n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
	
	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		reduceRowWise(deviceID, localSize, globalSize, input, output, n, sharedMemSize);
	}
};
)~~~";

std::string createReduce2DKernelProgram_CL(UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir)
{
	std::string rowKernelSource = ReduceKernelTemplate_CL;
	std::string colKernelSource = ReduceKernelTemplate_CL;
	std::stringstream sourceStream;
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + rowWiseFunc.uniqueName + "_" + colWiseFunc.uniqueName;
	
	
	if (rowWiseFunc.requiresDoublePrecision || colWiseFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
	
	replaceTextInString(rowKernelSource, PH_ReduceResultType, rowWiseFunc.resolvedReturnTypeName);
	replaceTextInString(rowKernelSource, PH_KernelName, kernelName + "_RowWise");
	replaceTextInString(rowKernelSource, PH_ReduceFuncName, rowWiseFunc.uniqueName);
	
	replaceTextInString(colKernelSource, PH_ReduceResultType, colWiseFunc.resolvedReturnTypeName);
	replaceTextInString(colKernelSource, PH_KernelName, kernelName + "_ColWise");
	replaceTextInString(colKernelSource, PH_ReduceFuncName, colWiseFunc.uniqueName);
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	std::set<UserType*> referencedUTs;
	std::set_union(
		rowWiseFunc.ReferencedUTs.cbegin(), rowWiseFunc.ReferencedUTs.cend(),
		colWiseFunc.ReferencedUTs.cbegin(), colWiseFunc.ReferencedUTs.cend(), std::inserter(referencedUTs, referencedUTs.begin()));
	
	for (UserType *RefType : referencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	const std::string className = "CLWrapperClass_" + kernelName;
	
	sourceStream << KernelPredefinedTypes_CL;
	
	if (rowWiseFunc.refersTo(colWiseFunc))
		sourceStream << generateUserFunctionCode_CL(rowWiseFunc);
	else if (colWiseFunc.refersTo(rowWiseFunc))
		sourceStream << generateUserFunctionCode_CL(colWiseFunc);
	else
		sourceStream << generateUserFunctionCode_CL(rowWiseFunc) << generateUserFunctionCode_CL(colWiseFunc);
	
	sourceStream << rowKernelSource << colKernelSource;
	
	std::string finalSource = Constructor2D;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	return kernelName;
}
