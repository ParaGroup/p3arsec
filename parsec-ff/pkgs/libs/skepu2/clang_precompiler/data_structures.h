#pragma once

#include <string>
#include <unordered_set>
#include <utility>

#include "clang/AST/AST.h"



// ------------------------------
// Data structures
// ------------------------------

enum class AccessMode
{
	Read, Write, ReadWrite
};

enum class ContainerType
{
	Vector, Matrix, SparseMatrix
};


struct Skeleton
{
	enum class Type
	{
		Map, Reduce1D, Reduce2D, MapReduce, Scan, MapOverlap1D, MapOverlap2D, Call
	};
	
	std::string name;
	Type type;
	size_t userfunctionArgAmount;
	size_t deviceKernelAmount;
};


class UserConstant
{
public:
	const clang::VarDecl *astDeclNode;
	
	std::string name;
	std::string typeName;
	std::string definition;
	
	UserConstant(const clang::VarDecl *v);
};


class UserType
{
public:
	const clang::CXXRecordDecl *astDeclNode;
	
	std::string name;
	bool requiresDoublePrecision;
	
	UserType(const clang::CXXRecordDecl *t);
};



class UserFunction
{
public:
	
	
	struct TemplateArgument
	{
		const std::string paramName;
		const std::string typeName;
		
		TemplateArgument(std::string name, std::string type);
	};
	
	struct Param
	{
		const clang::ParmVarDecl *astDeclNode;
		const clang::Type *type;
		
		std::string name;
		
		std::string rawTypeName;
		std::string resolvedTypeName;
		std::string escapedTypeName;
		std::string fullTypeName;
		
		virtual size_t numKernelArgsCL();
		
		Param(const clang::ParmVarDecl *p);
		
		virtual ~Param() = default;
		
		static bool constructibleFrom(const clang::ParmVarDecl *p);
	};
	
	struct RandomAccessParam: Param
	{
		AccessMode accessMode;
		ContainerType containerType;
		
		const clang::Type *containedType;
		
		std::string TypeNameOpenCL();
		std::string TypeNameHost();
		
		virtual size_t numKernelArgsCL() override;
		
		RandomAccessParam(const clang::ParmVarDecl *p);
		
		virtual ~RandomAccessParam() = default;
		
		static bool constructibleFrom(const clang::ParmVarDecl *p);
	};
	
	void updateArgLists(size_t arity);
	
	bool refersTo(UserFunction &other);
	
	std::string funcNameCUDA();
	
	clang::FunctionDecl *astDeclNode;
	
	std::string rawName;
	std::string uniqueName;
	std::string rawReturnTypeName;
	std::string resolvedReturnTypeName;
	
	std::string instanceName;
	
	clang::SourceLocation codeLocation;
	
	std::vector<Param> elwiseParams{};
	std::vector<RandomAccessParam> anyContainerParams {};
	std::vector<Param> anyScalarParams {};
	Param *indexParam; // can be NULL
	
	std::vector<TemplateArgument> templateArguments{};
	
	std::vector<std::pair<const clang::CallExpr*, UserFunction*>> UFReferences{};
	std::set<UserFunction*> ReferencedUFs{};
	
	std::vector<std::pair<const clang::TypeSourceInfo*, UserType*>> UTReferences{};
	std::set<UserType*> ReferencedUTs{};
	
	
	std::vector<clang::CXXOperatorCallExpr*> containerSubscripts{};
	
	bool fromTemplate = false;
	bool indexed1D = false;
	bool indexed2D = false;
	bool requiresDoublePrecision;
	
	size_t numKernelArgsCL();
	
	UserFunction(clang::FunctionDecl *f);
	UserFunction(clang::CXXMethodDecl *f, clang::VarDecl *d);
};

