#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/Lex/Lexer.h"
#include "clang/Basic/Diagnostic.h"

#include "data_structures.h"


extern llvm::cl::opt<bool> GenCUDA;
extern llvm::cl::opt<bool> GenOMP;
extern llvm::cl::opt<bool> GenCL;

extern llvm::cl::opt<std::string> ResultName;
extern llvm::cl::opt<std::string> ResultDir;

extern llvm::cl::opt<bool> Verbose;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::FunctionDecl*, UserFunction*> UserFunctions;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::TypeDecl*, UserType*> UserTypes;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::VarDecl*, UserConstant*> UserConstants;


extern const std::unordered_map<std::string, Skeleton> Skeletons;


extern clang::Rewriter GlobalRewriter;

//extern clang::SourceManager *SM;

const std::string SkePU_UF_Prefix {"skepu2_userfunction_"};


void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
std::string transformToCXXIdentifier(std::string &in);
