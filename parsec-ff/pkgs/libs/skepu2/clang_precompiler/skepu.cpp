#include "globals.h"
#include "visitor.h"

using namespace clang;

// ------------------------------
// Precompiler options
// ------------------------------

llvm::cl::OptionCategory SkepuPrecompilerCategory("SkePU source-to-source compiler");

llvm::cl::opt<bool> GenCUDA("cuda",  llvm::cl::desc("Generate CUDA backend"),   llvm::cl::cat(SkepuPrecompilerCategory));
llvm::cl::opt<bool> GenOMP("openmp", llvm::cl::desc("Generate OpenMP backend"), llvm::cl::cat(SkepuPrecompilerCategory));
llvm::cl::opt<bool> GenCL("opencl",  llvm::cl::desc("Generate OpenCL backend"), llvm::cl::cat(SkepuPrecompilerCategory));

llvm::cl::opt<bool> Verbose("verbose",  llvm::cl::desc("Verbose logging printout"), llvm::cl::cat(SkepuPrecompilerCategory));
llvm::cl::opt<bool> Silent("silent",  llvm::cl::desc("Disable normal printouts"), llvm::cl::cat(SkepuPrecompilerCategory));
llvm::cl::opt<bool> NoAddExtension("override-extension",  llvm::cl::desc("Do not automatically add file extension to output file (good for headers)"), llvm::cl::cat(SkepuPrecompilerCategory));

llvm::cl::opt<std::string> AllowedFuncNames("fnames", llvm::cl::desc("Function names which are allowed to be called from user functions"), llvm::cl::cat(SkepuPrecompilerCategory));

// Should be required
llvm::cl::opt<std::string> ResultDir("dir", llvm::cl::desc("Directory of output files"), llvm::cl::cat(SkepuPrecompilerCategory));
llvm::cl::opt<std::string> ResultName("name", llvm::cl::desc("File name of main output file (without extension, e.g., .cpp or .cu)"), llvm::cl::cat(SkepuPrecompilerCategory));

// Derived
static std::string mainFileName;


// ------------------------------
// Globals
// ------------------------------

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const FunctionDecl*, UserFunction*> UserFunctions;

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const TypeDecl*, UserType*> UserTypes;

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const VarDecl*, UserConstant*> UserConstants;

// Explicitly allowed functions to call from user functions
std::unordered_set<std::string> AllowedFunctionNamesCalledInUFs
{
	"sqrt",
	"abs",
	"printf",
	"pow",
};

// Skeleton types lookup from internal SkePU class template name
const std::unordered_map<std::string, Skeleton> Skeletons = 
{
	{"MapImpl",        {"Map",          Skeleton::Type::Map,          1, 1}},
	{"Reduce1D",       {"Reduce1D",     Skeleton::Type::Reduce1D,     1, 1}},
	{"Reduce2D",       {"Reduce2D",     Skeleton::Type::Reduce2D,     2, 2}},
	{"MapReduceImpl",  {"MapReduce",    Skeleton::Type::MapReduce,    2, 2}},
	{"ScanImpl",       {"Scan",         Skeleton::Type::Scan,         1, 3}},
	{"MapOverlap1D",   {"MapOverlap1D", Skeleton::Type::MapOverlap1D, 1, 4}},
	{"MapOverlap2D",   {"MapOverlap2D", Skeleton::Type::MapOverlap2D, 1, 1}},
	{"CallImpl",       {"Call",         Skeleton::Type::Call,         1, 1}},
};

Rewriter GlobalRewriter;


// For each source file provided to the tool, a new FrontendAction is created.
class SkePUFrontendAction : public ASTFrontendAction
{
public:
	
	bool BeginSourceFileAction(CompilerInstance &CI, StringRef Filename) override
	{
	//	if (Verbose) llvm::errs() << "** BeginSourceFileAction\n";
	//	SourceManager &SM = CI.getSourceManager();
		if (Verbose) llvm::errs() << "** BeginSourceFileAction for: " << Filename << "\n";
		
		return true;
	}
	
	void EndSourceFileAction() override
	{
		SourceManager &SM = GlobalRewriter.getSourceMgr();
		SourceLocation SLStart = SM.getLocForStartOfFile(SM.getMainFileID());
		
		GlobalRewriter.InsertText(SLStart, "#define SKEPU_PRECOMPILED\n");
		if (GenOMP)  GlobalRewriter.InsertText(SLStart, "#define SKEPU_OPENMP\n");
		if (GenCL)   GlobalRewriter.InsertText(SLStart, "#define SKEPU_OPENCL\n");
		if (GenCUDA) GlobalRewriter.InsertText(SLStart, "#define SKEPU_CUDA\n");
		
		for (VarDecl *d : this->SkeletonInstances)
			HandleSkeletonInstance(d);
		
		
		
		
		
		if (Verbose) llvm::errs() << "** EndSourceFileAction for: " << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";
		
		// Now emit the rewritten buffer.
		std::error_code EC;
		llvm::raw_fd_ostream OutFile(mainFileName, EC, llvm::sys::fs::F_RW);
		GlobalRewriter.getEditBuffer(SM.getMainFileID()).write(OutFile);
		OutFile.close();
	}
	
	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override
	{
		if (Verbose) llvm::errs() << "** Creating AST consumer for: " << file << "\n";
		GlobalRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<SkePUASTConsumer>(&CI.getASTContext(), this->SkeletonInstances);
	}
	
private:
	std::unordered_set<clang::VarDecl *> SkeletonInstances;
};


int main(int argc, const char **argv)
{
	tooling::CommonOptionsParser op(argc, argv, SkepuPrecompilerCategory);
	tooling::ClangTool Tool(op.getCompilations(), op.getSourcePathList());
	
	if (ResultName == "")
		ResultName = op.getSourcePathList()[0];
	mainFileName = ResultDir + "/" + ResultName + (NoAddExtension ? "" : (GenCUDA ? ".cu" : ".cpp"));
	
	if (!Silent)
	{
		llvm::errs() << "# ======================================= #\n";
		llvm::errs() << "~  SkePU source-to-source compiler v 0.1  ~\n";
		llvm::errs() << "# --------------------------------------- #\n";
		llvm::errs() << "   CUDA gen:\t" << (GenCUDA ? "ON" : "OFF") << "\n";
		llvm::errs() << "   OpenCL gen:\t" << (GenCL ? "ON" : "OFF") << "\n";
		llvm::errs() << "   OpenMP gen:\t" << (GenOMP ? "ON" : "OFF") << "\n";
		llvm::errs() << "   Main output file: " << mainFileName << "\n";
		llvm::errs() << "# ======================================= #\n";
	}
	
	std::istringstream SSNames(AllowedFuncNames);
	std::vector<std::string> Names{std::istream_iterator<std::string>{SSNames}, std::istream_iterator<std::string>{}};
	for (std::string &name : Names)
		AllowedFunctionNamesCalledInUFs.insert(name);
	
	return Tool.run(tooling::newFrontendActionFactory<SkePUFrontendAction>().get());
}
