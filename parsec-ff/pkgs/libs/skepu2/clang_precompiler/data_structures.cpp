#include <sstream>
#include <unordered_set>
#include <unordered_map>

#include "clang/Sema/Sema.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "globals.h"
#include "data_structures.h"
#include "visitor.h"

using namespace clang;
using namespace clang::ast_matchers;

// Defined in skepu.cpp
extern std::unordered_set<std::string> AllowedFunctionNamesCalledInUFs;


// This visitor traverses a userfunction AST node and finds references to other userfunctions and usertypes.
class UserFunctionVisitor : public RecursiveASTVisitor<UserFunctionVisitor>
{
public:
	
	bool VisitCallExpr(CallExpr *c)
	{
		if (isa<CXXOperatorCallExpr>(c))
			return true;
		
		if (auto *UnresolvedLookup = dyn_cast<UnresolvedLookupExpr>(c->getCallee()))
		{
			std::string name = UnresolvedLookup->getName().getAsString();
			if (Verbose) llvm::errs() << "Found unresolved lookup expr " << UnresolvedLookup->getName() <<"\n";
			
			bool allowed = std::find(AllowedFunctionNamesCalledInUFs.begin(), AllowedFunctionNamesCalledInUFs.end(), name)
				!= AllowedFunctionNamesCalledInUFs.end();
			if (!allowed)
				GlobalRewriter.getSourceMgr().getDiagnostics().Report(c->getLocStart(), diag::err_skepu_userfunction_call) << name;
			
			return allowed;
		}
		
		FunctionDecl *Func = c->getDirectCallee();
		std::string name = Func->getName();
			
		
		if (std::find(AllowedFunctionNamesCalledInUFs.begin(), AllowedFunctionNamesCalledInUFs.end(), name) != AllowedFunctionNamesCalledInUFs.end())
		{
			// Called function is explicitly allowed
			if (Verbose) llvm::errs() << "Found reference to allowed function: '" << name << "'\n";
		}
		else
		{
			if (Verbose) llvm::errs() << "Found reference to other userfunction: '" << name << "'\n";
			UserFunction *UF = HandleUserFunction(Func);
			UF->updateArgLists(0);
			UFReferences.emplace_back(c, UF);
			ReferencedUFs.insert(UF);
		}
		
		return true;
	}
	
	bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *c)
	{
		if (c->getOperator() == OO_Subscript)
			this->containerSubscripts.push_back(c);
		
		return true;
	}
	
	bool VisitVarDecl(VarDecl *d)
	{
		if (auto *userType = HandleUserType(d->getType().getTypePtr()->getAsCXXRecordDecl()))
			ReferencedUTs.insert(userType);
		
		return true;
	}
	
	
	std::vector<std::pair<const CallExpr*, UserFunction*>> UFReferences{};
	std::set<UserFunction*> ReferencedUFs{};
	
	std::vector<std::pair<const TypeSourceInfo*, UserType*>> UTReferences{};
	std::set<UserType*> ReferencedUTs{};
	
	std::vector<CXXOperatorCallExpr*> containerSubscripts{};
};



UserConstant::UserConstant(const VarDecl *v)
: astDeclNode(v)
{
	this->name = v->getNameAsString();
	this->typeName = v->getType().getAsString();
	
	SourceRange SRInit = v->getInit()->IgnoreImpCasts()->getSourceRange();
	this->definition = Lexer::getSourceText(CharSourceRange::getTokenRange(SRInit), GlobalRewriter.getSourceMgr(), LangOptions(), 0);
}


UserType::UserType(const CXXRecordDecl *t)
: astDeclNode(t), name(t->getNameAsString()), requiresDoublePrecision(false)
{
	
	if (const RecordDecl *r = dyn_cast<RecordDecl>(t))
	{
		for (const FieldDecl *f : r->fields())
		{
			std::string fieldName = f->getNameAsString();
			std::string typeName = f->getType().getAsString();
			if (Verbose) llvm::errs() << "User type '" << this->name << "': found field '" << fieldName << "' of type " << typeName << "\n";
			
			if (typeName == "double")
				this->requiresDoublePrecision = true;
		}
	}
	
	static const std::string RunTimeTypeNameFunc = R"~~~(
	namespace skepu2 { template<> std::string getDataTypeCL<TYPE_NAME>() { return "struct TYPE_NAME"; } }
	)~~~";
	
	if (GenCL)
	{
		std::string typeNameFunc = RunTimeTypeNameFunc;
		replaceTextInString(typeNameFunc, "TYPE_NAME", this->name);
		GlobalRewriter.InsertText(t->getLocEnd().getLocWithOffset(2), typeNameFunc);
	}
}

std::string random_string(size_t length)
{
	auto randchar = []() -> char
	{
		const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		const size_t max_index = sizeof(charset) - 1;
		return charset[ rand() % max_index ];
	};
	std::string str(length, 0);
	std::generate_n(str.begin(), length, randchar);
	return str;
}

UserFunction::Param::Param(const clang::ParmVarDecl *p)
: astDeclNode(p)
{
	this->name = p->getNameAsString();
	
	if (this->name == "")
		this->name = "unused_" + random_string(10);
	
	this->type = p->getOriginalType().getTypePtr();
	this->rawTypeName = p->getOriginalType().getAsString();
	this->resolvedTypeName = this->rawTypeName;
	this->escapedTypeName = transformToCXXIdentifier(this->resolvedTypeName);
	
	if (Verbose) llvm::errs() << "Param: " << this->name << " of type " << this->rawTypeName << " resolving to " << this->resolvedTypeName << "\n";
}

bool UserFunction::Param::constructibleFrom(const clang::ParmVarDecl *p)
{
	return !UserFunction::RandomAccessParam::constructibleFrom(p);
}

size_t UserFunction::Param::numKernelArgsCL()
{
	return 1;
}


UserFunction::RandomAccessParam::RandomAccessParam(const ParmVarDecl *p)
: Param(p)
{
	this->fullTypeName = this->resolvedTypeName;
	QualType underlying = p->getOriginalType();
	std::string qualifier;
	
	if (underlying.isConstQualified())
	{
		underlying = underlying.getUnqualifiedType();
		qualifier = "const";
		if (p->hasAttr<SkepuOutAttr>())
		{
			GlobalRewriter.getSourceMgr().getDiagnostics().Report(p->getAttr<SkepuOutAttr>()->getRange().getBegin(), diag::err_skepu_invalid_out_attribute) << this->name;
		}
		
		this->accessMode = AccessMode::Read;
		if (Verbose) llvm::errs() << "Read only access mode\n";
	}
	else if (p->hasAttr<SkepuOutAttr>())
	{
		this->accessMode = AccessMode::Write;
		if (Verbose) llvm::errs() << "Write only access mode\n";
	}
	else
	{
		this->accessMode = AccessMode::ReadWrite;
		if (Verbose) llvm::errs() << "ReadWrite access mode\n";
	}
	
	auto *type = underlying.getTypePtr();
	
	if (auto *innertype = dyn_cast<ElaboratedType>(type))
		type = innertype->getNamedType().getTypePtr();
	
	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	const clang::TemplateArgument containedTypeArg = templateType->getArg(0);
	
	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	this->containedType = containedTypeArg.getAsType().getTypePtr();
	this->rawTypeName = this->resolvedTypeName = containedTypeArg.getAsType().getAsString();
	this->escapedTypeName = transformToCXXIdentifier(this->resolvedTypeName);
	
	if (templateName == "SparseMat")
	{
		this->containerType = ContainerType::SparseMatrix;
		this->accessMode = AccessMode::Read; // Override for sparse matrices
		if (Verbose) llvm::errs() << "Sparse Matrix of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Mat")
	{
		this->containerType = ContainerType::Matrix;
		if (Verbose) llvm::errs() << "Matrix of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Vec")
	{
		this->containerType = ContainerType::Vector;
		if (Verbose) llvm::errs() << "Vector of " << this->resolvedTypeName << "\n";
	}
	else
		llvm::errs() << "FATAL ERROR\n";
	
	if (Verbose) llvm::errs() << "Param: " << this->name << " of type " << this->rawTypeName << " resolving to " << this->resolvedTypeName << " (or fully: " << this->fullTypeName << ")\n";
}

bool UserFunction::RandomAccessParam::constructibleFrom(const clang::ParmVarDecl *p)
{
	auto *type = p->getOriginalType().getTypePtr();
	if (auto *innertype = dyn_cast<ElaboratedType>(type))
		type = innertype->getNamedType().getTypePtr();
	
	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	if (!templateType) return false;
	
	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	return (templateName == "SparseMat") || (templateName == "Mat") || (templateName == "Vec");
}

std::string UserFunction::RandomAccessParam::TypeNameOpenCL()
{
	switch (this->containerType)
	{
		case ContainerType::Vector:
			return "skepu_vec_proxy_" + this->escapedTypeName;
		case ContainerType::Matrix:
			return "skepu_mat_proxy_" + this->escapedTypeName;
		case ContainerType::SparseMatrix:
			return "skepu_sparse_mat_proxy_" + this->escapedTypeName;
		default:
			llvm::errs() << "ERROR: TypeNameOpenCL: Invalid switch value\n";
			return "";
	}
}

std::string UserFunction::RandomAccessParam::TypeNameHost()
{
	switch (this->containerType)
	{
		case ContainerType::Vector:
			return "std::tuple<skepu2::Vector<" + this->resolvedTypeName + "> *, skepu2::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *>";
		case ContainerType::Matrix:
			return "std::tuple<skepu2::Matrix<" + this->resolvedTypeName + "> *, skepu2::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *>";
		case ContainerType::SparseMatrix:
			return "std::tuple<skepu2::SparseMatrix<" + this->resolvedTypeName + "> *, skepu2::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *, "
				+ "skepu2::backend::DeviceMemPointer_CL<size_t> *, skepu2::backend::DeviceMemPointer_CL<size_t> *>";
		default:
			llvm::errs() << "ERROR: TypeNameHost: Invalid switch value\n";
			return "";
	}
}

size_t UserFunction::RandomAccessParam::numKernelArgsCL()
{
	switch (this->containerType)
	{
		case ContainerType::Vector:
			return 2;
		case ContainerType::Matrix:
			return 3;
		case ContainerType::SparseMatrix:
			return 4;
		default:
			llvm::errs() << "ERROR: numKernelArgsCL: Invalid switch value\n";
			return 0;
	}
}




UserFunction::TemplateArgument::TemplateArgument(std::string name, std::string type)
: paramName(name), typeName(type)
{}


UserFunction::UserFunction(CXXMethodDecl *f, VarDecl *d)
: UserFunction(f)
{
	this->rawName = this->uniqueName = "lambda_uf_" + random_string(10);
	
	this->codeLocation = d->getSourceRange().getBegin();
	if (const FunctionDecl *DeclCtx = dyn_cast<FunctionDecl>(d->getDeclContext()))
		this->codeLocation = DeclCtx->getSourceRange().getBegin();
}

UserFunction::UserFunction(FunctionDecl *f)
: astDeclNode(f), requiresDoublePrecision(false)
{
	// Function name
	this->rawName = f->getNameInfo().getName().getAsString();
		
	// Code location
	this->codeLocation = f->getSourceRange().getEnd().getLocWithOffset(1);
	
	std::stringstream SSUniqueName;
	SSUniqueName << this->rawName;
	
	// Handle userfunction templates
	if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
	{
		this->fromTemplate = true;
		const FunctionTemplateDecl *t = f->getPrimaryTemplate();
		const TemplateParameterList *TPList = t->getTemplateParameters();
		const TemplateArgumentList *TAList = f->getTemplateSpecializationArgs(); 
		
		for (unsigned i = 0; i < TPList->size(); ++i)
		{
			std::string paramName = TPList->getParam(i)->getNameAsString();
			std::string argName = TAList->get(i).getAsType().getAsString();
			this->templateArguments.emplace_back(paramName, argName);
			SSUniqueName << "_" << transformToCXXIdentifier(argName);
			if (Verbose) llvm::errs() << "Template param: " << paramName << " = " << argName << "\n";
		}
	}
	
	this->uniqueName = SSUniqueName.str();
	if (Verbose) llvm::errs() << "Created UserFunction object with unique name '" << this->uniqueName << "'\n";
	
	if (!f->doesThisDeclarationHaveABody())
		GlobalRewriter.getSourceMgr().getDiagnostics().Report(f->getSourceRange().getEnd(), diag::err_skepu_no_userfunction_body) << this->rawName;
	
	// Type name as string
	this->rawReturnTypeName = f->getReturnType().getAsString();
	this->resolvedReturnTypeName = this->rawReturnTypeName;
	
	for (UserFunction::TemplateArgument &arg : templateArguments)
		if (this->rawReturnTypeName == arg.paramName)
			this->resolvedReturnTypeName = arg.typeName;
	
	// Argument lists
	auto it = f->param_begin();
	
	if (f->param_size() > 0)
	{
		std::string name = (*it)->getOriginalType().getAsString();
		this->indexed1D = (name == "skepu2::Index1D" || name == "struct skepu2::Index1D");
		this->indexed2D = (name == "skepu2::Index2D" || name == "struct skepu2::Index1D");
	}
	
	if (this->indexed1D || this->indexed2D)
		this->indexParam = new UserFunction::Param(*it++);
	
	// Referenced functions and types
	UserFunctionVisitor UFVisitor;
	UFVisitor.TraverseDecl(this->astDeclNode);
	
	this->ReferencedUFs = UFVisitor.ReferencedUFs;
	this->UFReferences = UFVisitor.UFReferences;
	
	this->ReferencedUTs = UFVisitor.ReferencedUTs;
	this->UTReferences = UFVisitor.UTReferences;
	
	this->containerSubscripts = UFVisitor.containerSubscripts;
	
	// Set requires double precision (TODO: more cases like parameters etc...)
	if (this->resolvedReturnTypeName == "double")
		this->requiresDoublePrecision = true; 
	
	for (UserType *UT : this->ReferencedUTs)
		if (UT->requiresDoublePrecision)
			this->requiresDoublePrecision = true; 
}

std::string UserFunction::funcNameCUDA()
{
	return SkePU_UF_Prefix + this->instanceName + "_" + this->uniqueName + "::CU";
}


size_t UserFunction::numKernelArgsCL()
{
	size_t count = 0;
	
	for (Param &p : this->elwiseParams)
		count += p.numKernelArgsCL();
	for (Param &p : this->anyContainerParams)
		count += p.numKernelArgsCL();
	for (Param &p : this->anyScalarParams)
		count += p.numKernelArgsCL();
	
	return count;
}

bool UserFunction::refersTo(UserFunction &other)
{
	// True if this is the other user function
	if (this == &other)
		return true;
	
	// True any of this's directly refered userfunctions refers to other
	for (auto *uf : this->ReferencedUFs)
		if (uf->refersTo(other))
			return true;
	
	return false;
}

void UserFunction::updateArgLists(size_t arity)
{
	if (Verbose) llvm::errs() << "Trying with arity: " << arity << "\n";
	
	this->elwiseParams.clear();
	this->anyContainerParams.clear();
	this->anyScalarParams.clear();
	
	auto it = this->astDeclNode->param_begin();
	const auto end = this->astDeclNode->param_end();
	
	if (this->indexed1D || this->indexed2D)
		it++;
	
	auto elwise_end = it + arity;
	while (it != end && it != elwise_end && UserFunction::Param::constructibleFrom(*it))
		this->elwiseParams.emplace_back(*it++);
	
	while (it != end && UserFunction::RandomAccessParam::constructibleFrom(*it))
		this->anyContainerParams.emplace_back(*it++);
	
	while (it != end)
		this->anyScalarParams.emplace_back(*it++);
	
	// Find references to user types
	auto scanForType = [&] (const Type *p)
	{
		if (auto *userType = HandleUserType(p->getAsCXXRecordDecl()))
			ReferencedUTs.insert(userType);
	};
	
	for (auto &param : this->elwiseParams)
		scanForType(param.type);
		
	for (auto &param : this->anyContainerParams)
		scanForType(param.containedType);
	
	for (auto &param : this->anyScalarParams)
		scanForType(param.type);
	
	if (Verbose)
	{
		llvm::errs() << "Deduced indexed: " << (this->indexParam ? "yes" : "no") << "\n";
		llvm::errs() << "Deduced elementwise arity: " << this->elwiseParams.size() << "\n";
		llvm::errs() << "Deduced random access arity: " << this->anyContainerParams.size() << "\n";
		llvm::errs() << "Deduced scalar arity: " << this->anyScalarParams.size() << "\n";
	}
}
