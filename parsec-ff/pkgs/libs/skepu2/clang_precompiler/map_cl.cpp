#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS __global SKEPU_MAP_RESULT_TYPE* output, size_t w, size_t n, size_t base)
{
	size_t i = get_global_id(0);
	size_t gridSize = get_local_size(0) * get_num_groups(0);
	SKEPU_CONTAINER_PROXIES
	
	while (i < n)
	{
		SKEPU_INDEX_INITIALIZER
		output[i] = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		i += gridSize;
	}
}
)~~~";


const std::string Constructor = R"~~~(
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
	
	static void map
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		SKEPU_HOST_KERNEL_PARAMS skepu2::backend::DeviceMemPointer_CL<SKEPU_MAP_RESULT_TYPE> *output,
		size_t w, size_t n, size_t base
	)
	{
		skepu2::backend::cl_helpers::setKernelArgs(kernels(deviceID), SKEPU_KERNEL_ARGS output->getDeviceDataPointer(), w, n, base);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
};
)~~~";


std::string createMapKernelProgram_CL(UserFunction &mapFunc, size_t arity, std::string dir)
{
	std::stringstream sourceStream, SSMapFuncParams, SSKernelParamList, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer;
	std::string indexInitializer;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
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
		SSHostKernelParamList << "skepu2::backend::DeviceMemPointer_CL<" << param.resolvedTypeName << "> *" << param.name << ", ";
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
				SSKernelParamList
					<< "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "__global size_t *" << param.name << "_row_pointers, "
					<< "__global size_t *" << param.name << "_col_indices, "
					<< "size_t skepu_size_" << param.name << ", ";
					
				SSKernelArgs
					<< "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<2>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<3>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->total_nnz(), ";
				
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { "
					<< ".data = " << name << ", "
					<< ".row_offsets = " << param.name << "_row_pointers, "
					<< ".col_indices = " << param.name << "_col_indices, "
					<< ".count = skepu_size_" << param.name << " };\n";
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
	
	if (mapFunc.requiresDoublePrecision)
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
	// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
	for (UserType *RefType : mapFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	std::stringstream SSKernelName;
	SSKernelName << transformToCXXIdentifier(ResultName) << "_MapKernel_" << mapFunc.uniqueName << "_arity_" << arity;
	const std::string kernelName = SSKernelName.str();
	const std::string className = "CLWrapperClass_" + kernelName;
	
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(mapFunc) << MapKernelTemplate_CL;
	
	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_MapResultType, mapFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_MapFuncName, mapFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, PH_MapParams, SSMapFuncParams.str());
	replaceTextInString(finalSource, PH_IndexInitializer, indexInitializer);
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}