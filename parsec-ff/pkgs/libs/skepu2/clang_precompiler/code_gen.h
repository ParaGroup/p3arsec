#pragma once

#include "globals.h"

extern llvm::cl::opt<std::string> ResultName;

const std::string PH_KernelName {"SKEPU_KERNEL_NAME"};
const std::string PH_KernelParams {"SKEPU_KERNEL_PARAMS"};

const std::string PH_MapFuncName {"SKEPU_FUNCTION_NAME_MAP"};
const std::string PH_MapOverlapFuncName {"SKEPU_FUNCTION_NAME_MAPOVERLAP"};
const std::string PH_ReduceFuncName {"SKEPU_FUNCTION_NAME_REDUCE"};
const std::string PH_ScanFuncName {"SKEPU_FUNCTION_NAME_SCAN"};
const std::string PH_CallFuncName {"SKEPU_FUNCTION_NAME_CALL"};

const std::string PH_MapResultType {"SKEPU_MAP_RESULT_TYPE"};
const std::string PH_ReduceResultType {"SKEPU_REDUCE_RESULT_TYPE"};
const std::string PH_ScanType {"SKEPU_SCAN_TYPE"};
const std::string PH_MapOverlapInputType {"SKEPU_MAPOVERLAP_INPUT_TYPE"};
const std::string PH_MapOverlapResultType {"SKEPU_MAPOVERLAP_RESULT_TYPE"};

const std::string PH_MapParams {"SKEPU_MAP_PARAMS"};
const std::string PH_ScanParams {"SKEPU_SCAN_PARAMS"};
const std::string PH_MapOverlapArgs {"SKEPU_MAPOVERLAP_ARGS"};
const std::string PH_CallArgs {"SKEPU_CALL_ARGS"};



const std::string PH_IndexInitializer {"SKEPU_INDEX_INITIALIZER"};


const std::string KernelPredefinedTypes_CL = R"~~~(
#define SKEPU_USING_BACKEND_CL 1

typedef struct{
	size_t i;
} index1_t;

typedef struct {
	size_t row;
	size_t col;
} index2_t;

size_t get_device_id()
{
	return SKEPU_INTERNAL_DEVICE_ID;
}

#define VARIANT_OPENCL(block) block
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block)

)~~~";

void generateUserFunctionStruct(UserFunction &UF, std::string InstanceName);
std::string generateOpenCLVectorProxy(std::string typeName);
std::string generateOpenCLMatrixProxy(std::string typeName);
std::string generateOpenCLSparseMatrixProxy(std::string typeName);
std::string replaceReferencesToOtherUFs(UserFunction &UF, std::function<std::string(UserFunction&)> nameFunc);
bool transformSkeletonInvocation(const Skeleton &skeleton, std::string InstanceName, std::vector<UserFunction*> FuncArgs, size_t arity, clang::VarDecl *d);

// CUDA generators
std::string createMapReduceKernelProgram_CU(UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir);
std::string createMapKernelProgram_CU(UserFunction &mapFunc, size_t arity, std::string dir);
std::string createScanKernelProgram_CU(UserFunction &scanFunc, std::string dir);
std::string createReduce1DKernelProgram_CU(UserFunction &reduceFunc, std::string dir);
std::string createReduce2DKernelProgram_CU(UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir);
std::string createMapOverlap1DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap2DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir);
std::string createCallKernelProgram_CU(UserFunction &callFunc, std::string dir);

// OpenCL helpers
std::string generateUserFunctionCode_CL(UserFunction &Func);
std::string generateUserTypeCode_CL(UserType &Type);

// OpenCL generators
std::string createMapReduceKernelProgram_CL(UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir);
std::string createMapKernelProgram_CL(UserFunction &mapFunc, size_t arity, std::string dir);
std::string createScanKernelProgram_CL(UserFunction &scanFunc, std::string dir);
std::string createReduce1DKernelProgram_CL(UserFunction &reduceFunc, std::string dir);
std::string createReduce2DKernelProgram_CL(UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir);
std::string createMapOverlap1DKernelProgram_CL(UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap2DKernelProgram_CL(UserFunction &mapOverlapFunc, std::string dir);
std::string createCallKernelProgram_CL(UserFunction &callFunc, std::string dir);
