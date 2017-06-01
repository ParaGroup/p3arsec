#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapReduceKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS __global SKEPU_REDUCE_RESULT_TYPE* output, size_t w, size_t n, size_t base, __local SKEPU_REDUCE_RESULT_TYPE* sdata)
{
	size_t blockSize = get_local_size(0);
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * blockSize + tid;
	size_t gridSize = blockSize * get_num_groups(0);
	SKEPU_REDUCE_RESULT_TYPE result = 0;
	SKEPU_CONTAINER_PROXIES
	
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
	{
		output[get_group_id(0)] = sdata[tid];
	}
}
)~~~";

static const char *ReduceKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME_ReduceOnly(__global SKEPU_REDUCE_RESULT_TYPE* input, __global SKEPU_REDUCE_RESULT_TYPE* output, size_t n, __local SKEPU_REDUCE_RESULT_TYPE* sdata)
{
	size_t blockSize = get_local_size(0);
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0)*blockSize + get_local_id(0);
	size_t gridSize = blockSize*get_num_groups(0);
	SKEPU_REDUCE_RESULT_TYPE result = 0;
	
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
	{
		output[get_group_id(0)] = sdata[tid];
	}
}
)~~~";


const std::string Constructor = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:
	
	enum
	{
		KERNEL_MAPREDUCE = 0,
		KERNEL_REDUCE,
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
			cl_kernel kernel_mapreduce = clCreateKernel(program, "SKEPU_KERNEL_NAME", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			cl_kernel kernel_reduce = clCreateKernel(program, "SKEPU_KERNEL_NAME_ReduceOnly", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			kernels(counter, KERNEL_MAPREDUCE, &kernel_mapreduce);
			kernels(counter, KERNEL_REDUCE,    &kernel_reduce);
			counter++;
		}
		
		initialized = true;
	}
	
	static void mapReduce
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		SKEPU_HOST_KERNEL_PARAMS
		skepu2::backend::DeviceMemPointer_CL<SKEPU_REDUCE_RESULT_TYPE> *output,
		size_t w, size_t n, size_t base,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MAPREDUCE);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, SKEPU_KERNEL_ARGS output->getDeviceDataPointer(), w, n, base);
		clSetKernelArg(kernel, SKEPU_KERNEL_ARG_COUNT + 4, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapReduce kernel");
	}
	
	static void reduceOnly
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_REDUCE_RESULT_TYPE> *input, skepu2::backend::DeviceMemPointer_CL<SKEPU_REDUCE_RESULT_TYPE> *output,
		size_t n, size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_REDUCE);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), output->getDeviceDataPointer(), n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapReduce reduce-only kernel");
	}
};
)~~~";


std::string createMapReduceKernelProgram_CL(UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapFuncParams, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	std::string indexInitializer;
	bool first = true;
	
	if (mapFunc.indexed1D)
	{
		SSMapFuncParams << "index";
		indexInitializer = "index1_t index = { .i = base + i };";
		first = false;
	}
	else if (mapFunc.indexed2D)
	{
		SSMapFuncParams << "index";
		indexInitializer = "index2_t index = { .row = (base + i) / w, .col = (base + i) % w };";
		first = false;
	}
	
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << "__global " << param.resolvedTypeName << " *" << param.name << ", ";
		SSHostKernelParamList << "skepu2::backend::DeviceMemPointer_CL<const " << param.resolvedTypeName << "> *" << param.name << ", ";
		SSKernelArgs << param.name << "->getDeviceDataPointer(), ";
		SSMapFuncParams << param.name << "[i]";
		first = false;
	}
	
	for (UserFunction::RandomAccessParam& param : mapFunc.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		if (!first) { SSMapFuncParams << ", "; }
		SSHostKernelParamList << param.TypeNameHost() << " skepu_container_" << param.name << ", ";
		containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
		switch (param.containerType)
		{
			case ContainerType::Vector:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->size(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name << ", .size = skepu_size_" << param.name << " };\n";
				break;
			case ContainerType::Matrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .rows = skepu_rows_" << param.name << ", .cols = skepu_cols_" << param.name << " };\n";
				break;
			case ContainerType::SparseMatrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "skepu_container_" << param.name << ".size(), ";
				break;
		}
		SSMapFuncParams << param.name;
		first = false;
	}
	
	for (UserFunction::Param& param : mapFunc.anyScalarParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSHostKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSKernelArgs << param.name << ", ";
		SSMapFuncParams << param.name;
		first = false;
	}
	
	assert(reduceFunc.elwiseParams.size() == 2);
	
	if (mapFunc.requiresDoublePrecision || reduceFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
	
	for (const std::string &type : containerProxyTypes[ContainerType::Vector])
		sourceStream << generateOpenCLVectorProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Matrix])
		sourceStream << generateOpenCLMatrixProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::SparseMatrix])
		sourceStream << generateOpenCLSparseMatrixProxy(type);
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	for (UserType *RefType : mapFunc.ReferencedUTs)
	{
		Rewriter R(GlobalRewriter.getSourceMgr(), LangOptions());
		sourceStream << R.getRewrittenText(RefType->astDeclNode->getSourceRange()) << ";\n\n";
	}
	
	std::stringstream SSKernelName;
	SSKernelName << transformToCXXIdentifier(ResultName) << "_MapReduceKernel_" << mapFunc.uniqueName << "_" << reduceFunc.uniqueName << "_arity_" << arity;
	const std::string kernelName = SSKernelName.str();
	const std::string className = "CLWrapperClass_" + kernelName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << mapFunc.numKernelArgsCL();
	
	sourceStream << KernelPredefinedTypes_CL;
	
	if (mapFunc.refersTo(reduceFunc))
		sourceStream << generateUserFunctionCode_CL(mapFunc);
	else if (reduceFunc.refersTo(mapFunc))
		sourceStream << generateUserFunctionCode_CL(reduceFunc);
	else
		sourceStream << generateUserFunctionCode_CL(mapFunc) << generateUserFunctionCode_CL(reduceFunc);
	
	sourceStream << MapReduceKernelTemplate_CL << ReduceKernelTemplate_CL;
	
	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARG_COUNT", SSKernelArgCount.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, PH_MapParams, SSMapFuncParams.str());
	replaceTextInString(finalSource, PH_ReduceResultType, reduceFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_MapResultType, mapFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_MapFuncName, mapFunc.uniqueName);
	replaceTextInString(finalSource, PH_ReduceFuncName, reduceFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_IndexInitializer, indexInitializer);
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}
