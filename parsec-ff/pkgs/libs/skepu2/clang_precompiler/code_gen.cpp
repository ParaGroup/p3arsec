#include "code_gen.h"

using namespace clang;


void printParamList(std::ostream &o, const FunctionDecl *f)
{
	bool first = true;
	for (const ParmVarDecl* Parm : f->params())
	{
		// default arg?
		if (!first) o << ", ";
		first = false;
		
		SourceRange SRParm = Parm->getSourceRange();
		std::string ParmText = Lexer::getSourceText(CharSourceRange::getTokenRange(SRParm), GlobalRewriter.getSourceMgr(), LangOptions(), 0);
		o << ParmText;
	}
}


void replaceTextInString(std::string& text, const std::string &find, const std::string &replace)
{
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos)
	{
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}

std::string transformToCXXIdentifier(std::string &in)
{
	std::string out = in;
	replaceTextInString(out, ".", "__dot__");
	replaceTextInString(out, " ", "__space__");
	return out;
}

static const std::string OpenCLVectorTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t size;
} skepu_vec_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLMatrixTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t rows;
	size_t cols;
} skepu_mat_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLSparseMatrixTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	__global size_t *row_offsets;
	__global size_t *col_indices;
	size_t count;
} skepu_sparse_mat_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

std::string generateOpenCLVectorProxy(std::string typeName)
{
	std::string retval = OpenCLVectorTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLMatrixProxy(std::string typeName)
{
	std::string retval = OpenCLMatrixTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLSparseMatrixProxy(std::string typeName)
{
	std::string retval = OpenCLSparseMatrixTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string replaceReferencesToOtherUFs(UserFunction &UF, std::function<std::string(UserFunction&)> nameFunc)
{
	const FunctionDecl *f = UF.astDeclNode;
	if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
		f = f->getTemplateInstantiationPattern();
	
// Find references to other userfunctions
	Rewriter R(GlobalRewriter.getSourceMgr(), LangOptions());
	
	for (auto &ref : UF.UFReferences)
		R.ReplaceText(ref.first->getCallee()->getSourceRange(), nameFunc(*ref.second));
		
	for (auto &ref : UF.UTReferences)
		R.ReplaceText(ref.first->getTypeLoc().getSourceRange(), "struct " + ref.second->name);
	
	for (auto subscript : UF.containerSubscripts)
		R.InsertText(subscript->getCallee()->getLocStart(), ".data");
	
	const CompoundStmt *Body = dyn_cast<CompoundStmt>(f->getBody());
	SourceRange SRBody = SourceRange(Body->getLocStart().getLocWithOffset(1), Body->getLocEnd().getLocWithOffset(-1));
	return R.getRewrittenText(SRBody);
}

bool testAndSet(bool &arg, bool newVal = true)
{
	bool oldVal = arg;
	arg = newVal;
	return oldVal;
}

void generateUserFunctionStruct(UserFunction &UF, std::string InstanceName)
{
	static std::set<std::string> generatedStructs;
	
	// Start by recursively generate functors for referenced user functions
	for (auto *referenced : UF.ReferencedUFs)
		generateUserFunctionStruct(*referenced, InstanceName);
	
	// Continue generating functor for this user function
	const FunctionDecl *f = UF.astDeclNode;
	std::string FunctorName = SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName;
	
	if (std::find(generatedStructs.begin(), generatedStructs.end(), FunctorName) != generatedStructs.end())
		return; // Already generated (same names implies idential code generated)
	
	generatedStructs.insert(FunctorName);
	UF.instanceName = InstanceName;
	
	std::stringstream SSSkepuFunctorStruct;
	SSSkepuFunctorStruct << "\nstruct " << FunctorName;
	SSSkepuFunctorStruct << "\n{\n";
	
	if (UF.fromTemplate)
		for (const UserFunction::TemplateArgument &arg : UF.templateArguments)
			SSSkepuFunctorStruct << "using " << arg.paramName << " = " << arg.typeName << ";\n";
	
	SSSkepuFunctorStruct << "constexpr static size_t totalArity = " << f->param_size() << ";\n";
	SSSkepuFunctorStruct << "constexpr static bool indexed = " << (UF.indexed1D || UF.indexed2D) << ";\n";
	
	SSSkepuFunctorStruct << "using ElwiseArgs = std::tuple<";
	bool first = true;
	for (UserFunction::Param& param : UF.elwiseParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		SSSkepuFunctorStruct << param.resolvedTypeName;
	}
	SSSkepuFunctorStruct << ">;\n";
	
	SSSkepuFunctorStruct << "using ContainerArgs = std::tuple<";
	first = true;
	for (UserFunction::Param& param : UF.anyContainerParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		SSSkepuFunctorStruct << param.fullTypeName;
	}
	SSSkepuFunctorStruct << ">;\n";
	
	SSSkepuFunctorStruct << "using UniformArgs = std::tuple<";
	first = true;
	for (UserFunction::Param& param : UF.anyScalarParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		SSSkepuFunctorStruct << param.resolvedTypeName;
	}
	SSSkepuFunctorStruct << ">;\n";
	
	SSSkepuFunctorStruct << "constexpr static skepu2::AccessMode anyAccessMode[] = {\n";
	
	for (auto param : UF.anyContainerParams)
	{
		SSSkepuFunctorStruct << "skepu2::AccessMode::";
		if (param.accessMode == AccessMode::Read)
			SSSkepuFunctorStruct << "Read, ";
		else if (param.accessMode == AccessMode::Write)
			SSSkepuFunctorStruct << "Write, ";
		else if (param.accessMode == AccessMode::ReadWrite)
			SSSkepuFunctorStruct << "ReadWrite, ";
	}
	SSSkepuFunctorStruct << "};\n\n";
	
	SSSkepuFunctorStruct << "using Ret = " << UF.resolvedReturnTypeName << ";\n\n";
	SSSkepuFunctorStruct << "constexpr static bool prefersMatrix = " << (UF.indexed2D) << ";\n\n";
	
	// CUDA code
	if (GenCUDA)
	{
		SSSkepuFunctorStruct << "#define SKEPU_USING_BACKEND_CUDA 1\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CPU\n";
		SSSkepuFunctorStruct << "#undef VARIANT_OPENMP\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CUDA\n";
		SSSkepuFunctorStruct << "#define VARIANT_CPU(block)\n";
		SSSkepuFunctorStruct << "#define VARIANT_OPENMP(block)\n";
		SSSkepuFunctorStruct << "#define VARIANT_CUDA(block) block\n";
		SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << "__device__ " << UF.resolvedReturnTypeName << " CU(";
		printParamList(SSSkepuFunctorStruct, f);
		SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::CU"; }) << "\n}\n";
		SSSkepuFunctorStruct << "#undef SKEPU_USING_BACKEND_CUDA\n\n";
	}
	
	if (GenOMP)
	{
		SSSkepuFunctorStruct << "#define SKEPU_USING_BACKEND_OMP 1\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CPU\n";
		SSSkepuFunctorStruct << "#undef VARIANT_OPENMP\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CUDA\n";
		SSSkepuFunctorStruct << "#define VARIANT_CPU(block)\n";
		SSSkepuFunctorStruct << "#define VARIANT_OPENMP(block) block\n";
		SSSkepuFunctorStruct << "#define VARIANT_CUDA(block)\n";
		SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << UF.resolvedReturnTypeName << " OMP(";
		printParamList(SSSkepuFunctorStruct, f);
		SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::OMP"; }) << "\n}\n";
		SSSkepuFunctorStruct << "#undef SKEPU_USING_BACKEND_OMP\n\n";
	}
	
	// CPU code
	SSSkepuFunctorStruct << "#define SKEPU_USING_BACKEND_CPU 1\n";
	SSSkepuFunctorStruct << "#undef VARIANT_CPU\n";
	SSSkepuFunctorStruct << "#undef VARIANT_OPENMP\n";
	SSSkepuFunctorStruct << "#undef VARIANT_CUDA\n";
	SSSkepuFunctorStruct << "#define VARIANT_CPU(block) block\n";
	SSSkepuFunctorStruct << "#define VARIANT_OPENMP(block)\n";
	SSSkepuFunctorStruct << "#define VARIANT_CUDA(block) block\n";
	SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << UF.resolvedReturnTypeName << " CPU(";
	printParamList(SSSkepuFunctorStruct, f);
	SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::CPU"; }) << "\n}\n";
	SSSkepuFunctorStruct << "#undef SKEPU_USING_BACKEND_CPU\n};\n\n";
	
	
	
	
	GlobalRewriter.InsertText(UF.codeLocation, SSSkepuFunctorStruct.str(), true, true);
} 

std::string getSourceAsString(SourceRange range)
{
	int rangeSize = GlobalRewriter.getRangeSize(range);
	if (rangeSize == -1)
		return "";
	
	SourceLocation startLoc = range.getBegin();
	const char *strStart = GlobalRewriter.getSourceMgr().getCharacterData(startLoc);
	
	std::string exprString;
	exprString.assign(strStart, rangeSize);
	
	return exprString;      
}

std::string generateUserTypeCode_CL(UserType &Type)
{
	std::string def = getSourceAsString(Type.astDeclNode->getSourceRange());
	return "typedef " + def + " " + Type.name + ";\n";
}

std::string generateUserFunctionCode_CL(UserFunction &Func)
{
	std::stringstream SSFuncParamList, SSFuncParams, SSFuncSource;
	bool first = true;
	
	if (Func.indexed1D)
	{
		SSFuncParamList << "index1_t " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed2D)
	{
		SSFuncParamList << "index2_t " << Func.indexParam->name;
		first = false;
	}
	
	for (UserFunction::Param& param : Func.elwiseParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		SSFuncParamList << param.resolvedTypeName << " " << param.name;
		first = false;
	}
	
	for (UserFunction::RandomAccessParam& param : Func.anyContainerParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		SSFuncParamList << param.TypeNameOpenCL() << " " << param.name;
		first = false;
	}
	
	for (UserFunction::Param& param : Func.anyScalarParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		if (param.astDeclNode->getOriginalType()->isPointerType())
			SSFuncParamList << "__global ";
		SSFuncParamList << param.resolvedTypeName << " " << param.name;
		first = false;
	}
	
	std::string transformedSource = replaceReferencesToOtherUFs(Func, [] (UserFunction &UF) { return UF.uniqueName; });
	
	// Recursive call, potential for circular loop: TODO FIX!!
	for (UserFunction *RefFunc : Func.ReferencedUFs)
		SSFuncSource << generateUserFunctionCode_CL(*RefFunc);
	
	SSFuncSource << "static " << Func.resolvedReturnTypeName << " " << Func.uniqueName << "(" << SSFuncParamList.str() << ")\n{";
	for (UserFunction::TemplateArgument &arg : Func.templateArguments)
		SSFuncSource << "typedef " << arg.typeName << " " << arg.paramName << ";\n";
		
	SSFuncSource << transformedSource << "\n}\n\n";
	return SSFuncSource.str();
}


bool transformSkeletonInvocation(const Skeleton &skeleton, std::string InstanceName, std::vector<UserFunction*> FuncArgs, size_t arity, VarDecl *d)
{
	if (Verbose) llvm::errs() << "Name of skeleton: " << skeleton.name << "\n";
	
	GlobalRewriter.RemoveText(d->getSourceRange());
	
	std::stringstream SSTemplateArgs, SSCallArgs, SSNewDecl;
	
	bool first = true;
	if (skeleton.type == Skeleton::Type::Map || skeleton.type == Skeleton::Type::MapReduce)
	{
		SSTemplateArgs << arity;
		first = false;
	}
	
	for (UserFunction *func : FuncArgs)
	{
		if (!first)
			SSTemplateArgs << ", ";
		SSTemplateArgs << SkePU_UF_Prefix << InstanceName << "_" << func->uniqueName;
		first = false;
	}
	
	if (GenCUDA)
	{
		std::string KernelName_CU;
		switch (skeleton.type)
		{
		case Skeleton::Type::MapReduce:
			KernelName_CU = createMapReduceKernelProgram_CU(*FuncArgs[0], *FuncArgs[1], arity, ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "), decltype(&" << KernelName_CU << "_ReduceOnly)";
			SSCallArgs << KernelName_CU << ", " << KernelName_CU << "_ReduceOnly";
			break;
			
		case Skeleton::Type::Map:
			KernelName_CU = createMapKernelProgram_CU(*FuncArgs[0], arity, ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
			
		case Skeleton::Type::Reduce1D:
			KernelName_CU = createReduce1DKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
			
		case Skeleton::Type::Reduce2D:
			KernelName_CU = createReduce2DKernelProgram_CU(*FuncArgs[0], *FuncArgs[1], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_RowWise), decltype(&" << KernelName_CU << "_ColWise)";
			SSCallArgs << KernelName_CU << "_RowWise, " << KernelName_CU << "_ColWise";
			break;
			
		case Skeleton::Type::Scan:
			KernelName_CU = createScanKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_ScanKernel), decltype(&" << KernelName_CU << "_ScanUpdate), decltype(&" << KernelName_CU << "_ScanAdd)";
			SSCallArgs << KernelName_CU << "_ScanKernel, " << KernelName_CU << "_ScanUpdate, " << KernelName_CU << "_ScanAdd";
			break;
			
		case Skeleton::Type::MapOverlap1D:
			KernelName_CU = createMapOverlap1DKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_MapOverlapKernel_CU), decltype(&" << KernelName_CU << "_MapOverlapKernel_CU_Matrix_Row), decltype(&"
				<< KernelName_CU << "_MapOverlapKernel_CU_Matrix_Col), decltype(&" << KernelName_CU << "_MapOverlapKernel_CU_Matrix_ColMulti)";
			SSCallArgs << KernelName_CU << "_MapOverlapKernel_CU, " << KernelName_CU << "_MapOverlapKernel_CU_Matrix_Row, "
				<< KernelName_CU << "_MapOverlapKernel_CU_Matrix_Col, " << KernelName_CU << "_MapOverlapKernel_CU_Matrix_ColMulti";
			break;
			
		case Skeleton::Type::MapOverlap2D:
			KernelName_CU = createMapOverlap2DKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_conv_cuda_2D_kernel)";
			SSCallArgs << KernelName_CU << "_conv_cuda_2D_kernel";
			break;
			
		case Skeleton::Type::Call:
			KernelName_CU = createCallKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
		}
		
		// Insert the code at the proper place
		const DeclContext *DeclCtx = d->getDeclContext();
		SourceLocation loc = d->getSourceRange().getBegin();
		if (isa<FunctionDecl>(DeclCtx))
		{
			loc = dyn_cast<FunctionDecl>(DeclCtx)->getSourceRange().getBegin();
		}
		GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CU + ".cu\"\n");
	}
	else
	{
		SSTemplateArgs << ", bool";
		SSCallArgs << "false";
		
		// Insert dummy arguments
		for (size_t i = 1; i < skeleton.deviceKernelAmount; ++i)
		{
			SSTemplateArgs << ", bool";
			SSCallArgs << ", false";
		}
	}
	
	if (GenCL)
	{
		std::string KernelName_CL;
		switch (skeleton.type)
		{
		case Skeleton::Type::MapReduce:
			KernelName_CL = createMapReduceKernelProgram_CL(*FuncArgs[0], *FuncArgs[1], arity, ResultDir);
			break;
			
		case Skeleton::Type::Map:
			KernelName_CL = createMapKernelProgram_CL(*FuncArgs[0], arity, ResultDir);
			break;
			
		case Skeleton::Type::Reduce1D:
			KernelName_CL = createReduce1DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::Reduce2D:
			KernelName_CL = createReduce2DKernelProgram_CL(*FuncArgs[0], *FuncArgs[1], ResultDir);
			break;
			
		case Skeleton::Type::Scan:
			KernelName_CL = createScanKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap1D:
			KernelName_CL = createMapOverlap1DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap2D:
			KernelName_CL = createMapOverlap2DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::Call:
			KernelName_CL = createCallKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
		}
		
		// Insert the code at the proper place
		SourceLocation loc = d->getSourceRange().getBegin();
		if (const FunctionDecl *DeclCtx = dyn_cast<FunctionDecl>(d->getDeclContext()))
			loc = DeclCtx->getSourceRange().getBegin();
		
		GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CL + "_cl_source.inl\"\n");
		
		SSTemplateArgs << ", CLWrapperClass_" << KernelName_CL;
	}
	else
	{
		SSTemplateArgs << ", void";
	}
	
	if (d->isStaticLocal())
		SSNewDecl << "static ";
	SSNewDecl << "skepu2::backend::" << skeleton.name << "<" << SSTemplateArgs.str() << "> " << InstanceName << "(" << SSCallArgs.str() << ")";
	
	GlobalRewriter.InsertText(d->getSourceRange().getBegin(), SSNewDecl.str());
	
	return true;
}
