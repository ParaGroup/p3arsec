#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS SKEPU_MAP_RESULT_TYPE *output, size_t w, size_t n, size_t base)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	while (i < n)
	{
		SKEPU_INDEX_INITIALIZER
		output[i] = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		i += gridSize;
	}
}
)~~~";


std::string createMapKernelProgram_CU(UserFunction &mapFunc, size_t arity, std::string dir)
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
	
	const std::string kernelName = ResultName + "_MapKernel_" + mapFunc.uniqueName;
	
	std::string kernelSource = MapKernelTemplate_CU;
	replaceTextInString(kernelSource, PH_MapResultType, mapFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_MapFuncName, mapFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(kernelSource, PH_MapParams, SSMapFuncParams.str());
	replaceTextInString(kernelSource, PH_IndexInitializer, indexInitializer);
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}