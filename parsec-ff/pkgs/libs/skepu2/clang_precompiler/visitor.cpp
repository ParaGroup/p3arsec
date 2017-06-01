#include "globals.h"
#include "code_gen.h"
#include "visitor.h"

using namespace clang;

std::unordered_set<std::string> SkeletonInstances;

// ------------------------------
// AST visitor
// ------------------------------

[[noreturn]] void SkePUAbort(std::string msg)
{
	llvm::errs() << "[SKEPU] INTERNAL FATAL ERROR: " << msg << "\n";
	exit(1);
}

UserFunction *HandleUserFunction(FunctionDecl *f)
{
	// Check so that this userfunction has not been treated already
	if (UserFunctions.find(f) != UserFunctions.end())
	{
		if (Verbose) llvm::errs() << "UF already handled!\n";
		return UserFunctions[f];
	}
	
	UserFunction *UF = new UserFunction(f); 
	UserFunctions[f] = UF;
	
	return UF;
}


UserFunction *HandleFunctionPointerArg(Expr *ArgExpr)
{
	// Check that the argument is a declaration reference
	if (!(DeclRefExpr::classof(ArgExpr)))
		SkePUAbort("Userfunction argument not a DeclRefExpr");
		
	// Get the referee delcaration of the argument declaration reference
	ValueDecl *ValDecl = dyn_cast<DeclRefExpr>(ArgExpr)->getDecl();
	
	// Check that the argument refers to a function declaration
	if (!FunctionDecl::classof(ValDecl))
		SkePUAbort("Userfunction argument not referencing a function declaration");
		
	// Get the function declaration and function name
	FunctionDecl *UserFunc = dyn_cast<FunctionDecl>(ValDecl);
	std::string FuncName = UserFunc->getNameInfo().getName().getAsString();
	
	return HandleUserFunction(UserFunc);
}

UserFunction *HandleLambdaArg(Expr *ArgExpr, VarDecl *d)
{
	CXXConstructExpr *ConstrExpr = dyn_cast<CXXConstructExpr>(ArgExpr);
	if (!ConstrExpr)
		SkePUAbort("User function (assumed lambda) argument not a construct expr");
	
	Expr *Arg = ConstrExpr->getArg(0);
	
	if (MaterializeTemporaryExpr *MatTempExpr = dyn_cast<MaterializeTemporaryExpr>(Arg))
		Arg = MatTempExpr->GetTemporaryExpr();
	
	LambdaExpr *Lambda = dyn_cast<LambdaExpr>(Arg);
	if (!Lambda)
		SkePUAbort("User function (assumed lambda) argument not a lambda");
	
	if (Lambda->capture_size() > 0)
		SkePUAbort("User function lambda argument has non-empty capture list");
	
	CXXMethodDecl *CallOperator = Lambda->getCallOperator();
	
	UserFunction *UF = new UserFunction(CallOperator, d); 
	UserFunctions[CallOperator] = UF;
	
	return UF;
}


const Skeleton::Type* DeclIsValidSkeleton(VarDecl *d)
{
	if (isa<ParmVarDecl>(d))
		return nullptr;
	
	if (d->isThisDeclarationADefinition() != VarDecl::DefinitionKind::Definition)
		return nullptr;
	
	Expr *InitExpr = d->getInit();
	if (!InitExpr)
		return nullptr;
	
	if (auto *CleanUpExpr = dyn_cast<ExprWithCleanups>(InitExpr))
		InitExpr = CleanUpExpr->getSubExpr();
	
	auto *ConstructExpr = dyn_cast<CXXConstructExpr>(InitExpr);
	if (!ConstructExpr || ConstructExpr->getConstructionKind() != CXXConstructExpr::ConstructionKind::CK_Complete)
		return nullptr;
	
	if (ConstructExpr->getNumArgs() == 0)
		return nullptr;
	
	auto *TempExpr = ConstructExpr->getArgs()[0];
	
	if (auto *MatTempExpr = dyn_cast<MaterializeTemporaryExpr>(TempExpr))
		TempExpr = MatTempExpr->GetTemporaryExpr();
	
	if (auto *BindTempExpr = dyn_cast<CXXBindTemporaryExpr>(TempExpr))
		TempExpr = BindTempExpr->getSubExpr();
	
	CallExpr *CExpr = dyn_cast<CallExpr>(TempExpr);
	if (!CExpr)
		return nullptr;
	
	const FunctionDecl *Callee = CExpr->getDirectCallee();
	const Type *RetType = Callee->getReturnType().getTypePtr();
	
	if (isa<DecltypeType>(RetType))
		RetType = dyn_cast<DecltypeType>(RetType)->getUnderlyingType().getTypePtr();
	
	if (auto *ElabType = dyn_cast<ElaboratedType>(RetType))
		RetType = ElabType->getNamedType().getTypePtr();
	
	if (!isa<TemplateSpecializationType>(RetType))
		return nullptr;
	
	const TemplateDecl *Template = RetType->getAs<TemplateSpecializationType>()->getTemplateName().getAsTemplateDecl();
	std::string TypeName = Template->getNameAsString();
	
	if (Skeletons.find(TypeName) == Skeletons.end())
		return nullptr;
	
	return &Skeletons.at(TypeName).type;
}

bool HandleSkeletonInstance(VarDecl *d)
{
	if (d->isThisDeclarationADefinition() != VarDecl::DefinitionKind::Definition)
		SkePUAbort("Not a definition");
	
	Expr *InitExpr = d->getInit();
	
	if (isa<ExprWithCleanups>(InitExpr))
		InitExpr = dyn_cast<ExprWithCleanups>(InitExpr)->getSubExpr();
	
	CXXConstructExpr *ConstructExpr = dyn_cast<CXXConstructExpr>(InitExpr);
	if (!ConstructExpr || ConstructExpr->getConstructionKind() != CXXConstructExpr::ConstructionKind::CK_Complete)
		SkePUAbort("Not a complete constructor");
		
	Expr *TempExpr = dyn_cast<MaterializeTemporaryExpr>(ConstructExpr->getArgs()[0])->GetTemporaryExpr();
	
	 if (isa<CXXBindTemporaryExpr>(TempExpr))
		TempExpr = dyn_cast<CXXBindTemporaryExpr>(TempExpr)->getSubExpr();
	
	CallExpr *CExpr = dyn_cast<CallExpr>(TempExpr);
	if (!CExpr)
		SkePUAbort("Not a call expression");
	
	
	const FunctionDecl *Callee = CExpr->getDirectCallee();
	const Type *RetType = Callee->getReturnType().getTypePtr();
	
	if (isa<DecltypeType>(RetType))
		RetType = dyn_cast<DecltypeType>(RetType)->getUnderlyingType().getTypePtr();
	
	const TemplateDecl *Template = RetType->getAs<TemplateSpecializationType>()->getTemplateName().getAsTemplateDecl();
	std::string TypeName = Template->getNameAsString();
	Skeleton::Type skeletonType = Skeletons.at(TypeName).type;
	
	std::string InstanceName = d->getNameAsString();
	SkeletonInstances.insert(InstanceName);
	
	size_t arity[2] = { 0, 2 };
	switch (skeletonType)
	{
	case Skeleton::Type::Map:
	case Skeleton::Type::MapReduce:
		assert(Callee->getTemplateSpecializationArgs()->size() > 0);
		arity[0] = Callee->getTemplateSpecializationArgs()->get(0).getAsIntegral().getExtValue();
		break;
	case Skeleton::Type::MapOverlap1D:
		arity[0] = 3; break;
	case Skeleton::Type::MapOverlap2D:
		arity[0] = 4; break;
	default:
		break;
	}
	
	std::vector<UserFunction*> FuncArgs;
	size_t i = 0;
	for (Expr *expr : CExpr->arguments())
	{
		UserFunction *UF;
		
		// The argument may be an implcit cast, get the underlying expression
		if (ImplicitCastExpr *ImplExpr = dyn_cast<ImplicitCastExpr>(expr))
			UF = HandleFunctionPointerArg(ImplExpr->IgnoreImpCasts());
		
		// It can also be an explicit cast, get the underlying expression
		else if (UnaryOperator *UnaryCastExpr = dyn_cast<UnaryOperator>(expr))
			UF = HandleFunctionPointerArg(UnaryCastExpr->getSubExpr());
		
		// The user function is probably defined as a lambda
		else
			UF = HandleLambdaArg(expr, d);
		
		FuncArgs.push_back(UF);
		UF->updateArgLists(arity[i++]);
		generateUserFunctionStruct(*UF, InstanceName);
	}
	
	return transformSkeletonInvocation(Skeletons.at(TypeName), InstanceName, FuncArgs, arity[0], d);
}

// Returns nullptr if the user type can be ignored
UserType *HandleUserType(const CXXRecordDecl *t)
{
	if (!t)
		return nullptr;
	
	std::string name = t->getNameAsString();
	if (Verbose) llvm::errs() << "Found user type: " << name << "\n";
	
	// These types are handled separately
	if (name == "Index1D" || name == "Index2D" || name == "Vec" || name == "Mat")
		return nullptr;
	
	if (!t->isCLike())
		SkePUAbort("User type is not C-like and not supported (run verbose for more details)\n");
	
	// Check if already handled, otherwise construct and add
	if (UserTypes.find(t) == UserTypes.end())
		UserTypes[t] = new UserType(t);
	
	return UserTypes[t];
}



SkePUASTVisitor::SkePUASTVisitor(ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet)
: SkeletonInstances(instanceSet), Context(ctx)
{}

bool SkePUASTVisitor::VisitVarDecl(VarDecl *d)
{
	if (!this->Context->getSourceManager().isInMainFile(d->getLocStart()))
		return RecursiveASTVisitor<SkePUASTVisitor>::VisitVarDecl(d);
	
	// Change this condition to check for skeleon class names, (and namespace too?)
//	if (d->hasAttr<SkepuInstanceAttr>())
	if (DeclIsValidSkeleton(d))
	{
		if (Verbose) llvm::errs() << "Found instance: " << d->getNameAsString() << "\n";
		
		SkeletonInstances.insert(d);
	}
	else if (d->hasAttr<SkepuUserConstantAttr>())
	{
		if (Verbose) llvm::errs() << "Found user constant: " << d->getNameAsString() << "\n";
		if (!(d->isConstexpr() && d->hasDefinition() == VarDecl::Definition))
		{
			llvm::errs() << "Invalid!\n"; // TODO: diagnostic
		}
		UserConstants[d] = new UserConstant(d);
	}
	return RecursiveASTVisitor<SkePUASTVisitor>::VisitVarDecl(d);
}


// Implementation of the ASTConsumer interface for reading an AST produced by the Clang parser.
SkePUASTConsumer::SkePUASTConsumer(ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet)
: Visitor(ctx, instanceSet)
{}

// Override the method that gets called for each parsed top-level
// declaration.
bool SkePUASTConsumer::HandleTopLevelDecl(DeclGroupRef DR)
{
	for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b)
	{
		// Traverse the declaration using our AST visitor.
		Visitor.TraverseDecl(*b);
	}
	return true;
}
