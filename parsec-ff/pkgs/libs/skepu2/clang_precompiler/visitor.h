#include "globals.h"
#include "code_gen.h"

// ------------------------------
// AST visitor
// ------------------------------

UserFunction *HandleUserFunction(clang::FunctionDecl *f);
UserType *HandleUserType(const clang::CXXRecordDecl *t);
bool HandleSkeletonInstance(clang::VarDecl *d);




class SkePUASTVisitor : public clang::RecursiveASTVisitor<SkePUASTVisitor>
{
public:
	
	SkePUASTVisitor(clang::ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet);
	
	bool VisitVarDecl(clang::VarDecl *d);
	
	std::unordered_set<clang::VarDecl *> &SkeletonInstances;

private:
	clang::ASTContext *Context;
};

// Implementation of the ASTConsumer interface for reading an AST produced by the Clang parser.
class SkePUASTConsumer : public clang::ASTConsumer
{
public:
	
	SkePUASTConsumer(clang::ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet);
	
	// Override the method that gets called for each parsed top-level declaration.
	bool HandleTopLevelDecl(clang::DeclGroupRef DR) override;

private:
	SkePUASTVisitor Visitor;
	
};
